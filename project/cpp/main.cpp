// micrograd_cli: trains a small MLP on 2D data.
// Reads one JSON request on stdin, writes one JSON response on stdout.
// The protocol is described in CLAUDE.md ("CLI protocol"); python/cli.py speaks the same one.

#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "micrograd.hpp"
#include "third_party/json.hpp"

using namespace std;
using json = nlohmann::json;

const int MAX_POINTS = 400;   // build_topo is recursive: keep the graph shallow

// Thrown when the loss becomes NaN/inf. Separate from bad-input errors so the message is clear.
struct Diverged : runtime_error {
    explicit Diverged(int epoch)
        : runtime_error("training diverged at epoch " + to_string(epoch) + ", try a lower learning rate") {}
};

// Everything the request asks for, after validation.
struct Request {
    Activation activation = Activation::Tanh;
    vector<int> layers;
    int epochs = 0;
    double lr = 0.0;
    vector<double> xs, ys, labels;
    bool has_init_weights = false;
    vector<double> init_weights;
    unsigned seed = 1;
    int snapshot_every = 0;
    int nx = 0, ny = 0;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
};

// ---------------------------------------------------------------------------
// Input validation helpers: each one throws invalid_argument with a readable message.
// ---------------------------------------------------------------------------

// `name` is a const char* (not a string) so callers don't create a temporary string;
// g++ would otherwise warn that the returned reference might dangle (it doesn't, but zero warnings).
const json& field(const json& obj, const char* name) {
    if(!obj.contains(name)) throw invalid_argument("missing field '" + string(name) + "'");
    return obj.at(name);
}

double get_number(const json& j, const string& name) {
    if(!j.is_number()) throw invalid_argument("'" + name + "' must be a number");
    double v = j.get<double>();
    if(!isfinite(v)) throw invalid_argument("'" + name + "' must be finite");
    return v;
}

int get_int(const json& j, const string& name, int min_value) {
    if(!j.is_number_integer()) throw invalid_argument("'" + name + "' must be an integer");
    long long v = j.get<long long>();
    if(v < min_value) throw invalid_argument("'" + name + "' must be >= " + to_string(min_value));
    if(v > 1000000000LL) throw invalid_argument("'" + name + "' is too large");
    return static_cast<int>(v);
}

Request parse_request(const json& req) {
    if(!req.is_object()) throw invalid_argument("request must be a JSON object");
    Request r;

    const json& act = field(req, "activation");
    if(!act.is_string()) throw invalid_argument("'activation' must be a string");
    string name = act.get<string>();
    if(name == "tanh") r.activation = Activation::Tanh;
    else if(name == "relu") r.activation = Activation::ReLU;
    else if(name == "sigmoid") r.activation = Activation::Sigmoid;
    else throw invalid_argument("'activation' must be one of tanh, relu, sigmoid");

    const json& layers = field(req, "layers");
    if(!layers.is_array() || layers.size() < 2)
        throw invalid_argument("'layers' must be an array with at least 2 sizes");
    for(const auto& n : layers) r.layers.push_back(get_int(n, "layers[i]", 1));
    if(r.layers.front() != 2) throw invalid_argument("'layers' must start with 2 (the data is 2D)");
    if(r.layers.back() != 1) throw invalid_argument("'layers' must end with 1 (one output)");

    r.epochs = get_int(field(req, "epochs"), "epochs", 1);
    r.lr = get_number(field(req, "lr"), "lr");

    const json& data = field(req, "data");
    if(!data.is_array() || data.empty()) throw invalid_argument("'data' must be a non-empty array");
    if(data.size() > MAX_POINTS)
        throw invalid_argument("too many points: " + to_string(data.size())
                               + " (max " + to_string(MAX_POINTS) + ")");
    for(const auto& p : data) {
        if(!p.is_array() || p.size() != 3) throw invalid_argument("each data point must be [x, y, label]");
        double label = get_number(p[2], "label");
        if(label != -1.0 && label != 1.0) throw invalid_argument("labels must be -1 or 1");
        r.xs.push_back(get_number(p[0], "x"));
        r.ys.push_back(get_number(p[1], "y"));
        r.labels.push_back(label);
    }

    if(req.contains("init_weights") && !req.at("init_weights").is_null()) {
        const json& w = req.at("init_weights");
        if(!w.is_array()) throw invalid_argument("'init_weights' must be an array");
        for(const auto& v : w) r.init_weights.push_back(get_number(v, "init_weights[i]"));
        r.has_init_weights = true;   // the length is checked against the network in main()
    }

    if(req.contains("seed")) r.seed = static_cast<unsigned>(get_int(req.at("seed"), "seed", 0));
    if(req.contains("snapshot_every")) r.snapshot_every = get_int(req.at("snapshot_every"), "snapshot_every", 0);

    if(r.snapshot_every > 0) {
        const json& g = field(req, "grid");
        if(!g.is_object()) throw invalid_argument("'grid' must be an object");
        r.nx = get_int(field(g, "nx"), "grid.nx", 1);
        r.ny = get_int(field(g, "ny"), "grid.ny", 1);
        r.xmin = get_number(field(g, "xmin"), "grid.xmin");
        r.xmax = get_number(field(g, "xmax"), "grid.xmax");
        r.ymin = get_number(field(g, "ymin"), "grid.ymin");
        r.ymax = get_number(field(g, "ymax"), "grid.ymax");
        if(r.xmin >= r.xmax || r.ymin >= r.ymax) throw invalid_argument("grid needs xmin < xmax and ymin < ymax");
    }

    return r;
}

// ---------------------------------------------------------------------------
// Training helpers
// ---------------------------------------------------------------------------

// Sum over samples of (pred - label)^2. Builds the whole graph, so loss.backward() works.
Value compute_loss(MLP& net, const vector<vector<Value>>& inputs, const vector<double>& labels) {
    Value loss(0.0);
    for(size_t i = 0; i < inputs.size(); i++) {
        Value pred = net(inputs[i])[0];
        Value diff = pred - Value(labels[i]);
        loss = loss + diff * diff;
    }
    return loss;
}

double round3(double v) { return std::round(v * 1000.0) / 1000.0; }

// One animation frame: the prediction at the centre of every grid cell, row j = 0 at ymin.
json make_frame(MLP& net, const Request& r, int epoch, double loss) {
    double dx = (r.xmax - r.xmin) / r.nx;
    double dy = (r.ymax - r.ymin) / r.ny;
    vector<double> grid;
    grid.reserve(static_cast<size_t>(r.nx) * r.ny);
    for(int j = 0; j < r.ny; j++) {
        for(int i = 0; i < r.nx; i++) {
            double x = r.xmin + (i + 0.5) * dx;
            double y = r.ymin + (j + 0.5) * dy;
            grid.push_back(round3(net({Value(x), Value(y)})[0].data()));
        }
    }
    return json{{"epoch", epoch}, {"loss", loss}, {"grid", grid}};
}

json run(const Request& r) {
    seed_rng(r.seed);
    vector<int> nouts(r.layers.begin() + 1, r.layers.end());
    MLP net(r.layers[0], nouts, r.activation);
    if(r.has_init_weights) net.set_parameters(r.init_weights);   // throws on a wrong length

    vector<vector<Value>> inputs;
    for(size_t i = 0; i < r.xs.size(); i++) inputs.push_back({Value(r.xs[i]), Value(r.ys[i])});

    auto params = net.parameters();
    vector<double> loss_history;
    json frames = json::array();
    double train_seconds = 0.0;

    // Only the training steps are timed: the clock is paused while a frame is computed.
    auto start = chrono::steady_clock::now();
    for(int epoch = 1; epoch <= r.epochs; epoch++) {
        Value loss = compute_loss(net, inputs, r.labels);
        if(!isfinite(loss.data())) throw Diverged(epoch);
        loss_history.push_back(loss.data());

        // Intermediate frames use the pre-update weights, so the grid matches this epoch's loss.
        // The last epoch's frame is made after training, from the final weights.
        bool want_frame = r.snapshot_every > 0 && epoch != r.epochs
                          && (epoch == 1 || epoch % r.snapshot_every == 0);
        if(want_frame) {
            train_seconds += chrono::duration<double>(chrono::steady_clock::now() - start).count();
            frames.push_back(make_frame(net, r, epoch, loss.data()));
            start = chrono::steady_clock::now();
        }

        // zero grads before backward, otherwise they accumulate
        for(auto& p : params) p.node->grad = 0.0;
        loss.backward();

        // gradient descent
        for(auto& p : params) p.node->data -= r.lr * p.node->grad;
    }
    train_seconds += chrono::duration<double>(chrono::steady_clock::now() - start).count();

    // One fresh forward pass with the final weights: final_loss, accuracy and the last frame.
    double final_loss = 0.0;
    int correct = 0;
    for(size_t i = 0; i < inputs.size(); i++) {
        double pred = net(inputs[i])[0].data();
        double diff = pred - r.labels[i];
        final_loss += diff * diff;
        double guess = (pred >= 0) ? 1.0 : -1.0;
        if(guess == r.labels[i]) correct++;
    }
    if(!isfinite(final_loss)) throw Diverged(r.epochs);
    if(r.snapshot_every > 0) frames.push_back(make_frame(net, r, r.epochs, final_loss));

    return json{
        {"ok", true},
        {"impl", "cpp"},
        {"param_count", params.size()},
        {"train_seconds", train_seconds},
        {"final_loss", final_loss},
        {"accuracy", static_cast<double>(correct) / inputs.size()},
        {"loss_history", loss_history},
        {"frames", frames},
    };
}

int fail(const string& message) {
    cout << json{{"ok", false}, {"error", message}}.dump() << "\n";
    return 1;
}

int main() {
    // istreambuf_iterator reads stdin byte by byte until EOF, whitespace included.
    string input((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());

    json req;
    try {
        req = json::parse(input);
    } catch(const json::parse_error&) {
        return fail("request is not valid JSON");
    }

    try {
        Request r = parse_request(req);
        cout << run(r).dump() << "\n";
        return 0;
    } catch(const exception& e) {
        return fail(e.what());
    }
}
