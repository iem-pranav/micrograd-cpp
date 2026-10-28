// micrograd in C++
// A port of Andrej Karpathy's micrograd (scalar autograd engine + tiny MLP).
// Build:  g++ -std=c++17 -Wall -Wextra -O2 micrograd.cpp -o micrograd
// Run:    ./micrograd [tanh|relu|sigmoid] [seed]

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include <cassert>
using namespace std;

// ---------------------------------------------------------------------------
// Autograd engine
// ---------------------------------------------------------------------------

struct Node {
    double data;
    double grad = 0.0;
    vector<shared_ptr<Node>> prev;   // children (the inputs that produced this node)
    string op;
    function<void()> backward_fn = [](){};  // like Python's `lambda: None`

    Node(double d, vector<shared_ptr<Node>> children = {}, string o = "")
        : data(d), prev(move(children)), op(move(o)) {}
};

void build_topo(Node* v, unordered_set<Node*>& visited, vector<Node*>& topo){
    if(visited.count(v) == 0){
        visited.insert(v);
        for(auto child : v->prev) build_topo(child.get(), visited, topo);
        topo.push_back(v);
    }
}

// Value is a cheap handle: copying a Value copies the pointer, not the Node,
// so every copy refers to the same data/grad.
// The constructor is intentionally not `explicit`, so `a + 2.0` and `2.0 + a` work.
class Value {
public:
    shared_ptr<Node> node;

    Value(double d = 0.0) {
        node = make_shared<Node>(d);
    }

    double data() const { return node->data; }
    double grad() const { return node->grad; }

    void backward() {
        unordered_set<Node*> visited;
        vector<Node*> topo;
        node->grad = 1.0;
        build_topo(node.get(), visited, topo);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) (*it)->backward_fn();
    }
};

// Each op's closure keeps its children alive (an, bn) but only holds a raw
// pointer (o) to its own node: otherwise node -> closure -> node would be a
// shared_ptr cycle and nothing would ever be freed.

Value operator+(const Value& a, const Value& b) {
    Value out(a.data() + b.data());
    out.node->prev = {a.node, b.node};
    out.node->op = "+";

    Node* o = out.node.get();
    auto an = a.node;
    auto bn = b.node;

    out.node->backward_fn = [an, bn, o]() {
        an->grad += 1 * o->grad;
        bn->grad += 1 * o->grad;
    };

    return out;
}

Value operator-(const Value& a, const Value& b) {
    Value out(a.data() - b.data());
    out.node->prev = {a.node, b.node};
    out.node->op = "-";

    Node* o = out.node.get();
    auto an = a.node;
    auto bn = b.node;

    out.node->backward_fn = [an, bn, o]() {
        an->grad += 1 * o->grad;
        bn->grad += -1 * o->grad;
    };

    return out;
}

Value operator-(const Value& a) {
    Value out(-1 * a.data());
    out.node->prev = {a.node};
    out.node->op = "neg";

    Node* o = out.node.get();
    auto an = a.node;

    out.node->backward_fn = [an, o]() {
        an->grad += -1 * o->grad;
    };

    return out;
}

Value operator*(const Value& a, const Value& b) {
    Value out(a.data() * b.data());
    out.node->prev = {a.node, b.node};
    out.node->op = "*";

    Node* o = out.node.get();
    auto an = a.node;
    auto bn = b.node;

    out.node->backward_fn = [an, bn, o]() {
        an->grad += bn->data * o->grad;
        bn->grad += an->data * o->grad;
    };

    return out;
}

Value pow(const Value& a, double p) {
    Value out(std::pow(a.data(), p));
    out.node->prev = {a.node};
    out.node->op = "^";

    Node* o = out.node.get();
    auto an = a.node;

    out.node->backward_fn = [an, p, o]() {
        an->grad += (p * std::pow(an->data, p - 1)) * o->grad;
    };

    return out;
}

Value operator/(const Value& a, const Value& b) {
    Value out(a.data() / b.data());
    out.node->prev = {a.node, b.node};
    out.node->op = "/";

    Node* o = out.node.get();
    auto an = a.node;
    auto bn = b.node;

    out.node->backward_fn = [an, bn, o]() {
        an->grad += std::pow(bn->data, -1) * o->grad;
        bn->grad += (-an->data * std::pow(bn->data, -2)) * o->grad;
    };

    return out;
}

Value exp(const Value& a) {
    Value out(std::exp(a.data()));
    out.node->prev = {a.node};
    out.node->op = "exp";

    Node* o = out.node.get();
    auto an = a.node;

    out.node->backward_fn = [an, o]() {
        an->grad += o->data * o->grad;
    };

    return out;
}

Value relu(const Value& a) {
    Value out(std::max(0.0, a.data()));
    out.node->prev = {a.node};
    out.node->op = "relu";

    Node* o = out.node.get();
    auto an = a.node;

    out.node->backward_fn = [an, o]() {
        if(an->data > 0) an->grad += 1 * o->grad;
        else an->grad += 0 * o->grad;
    };

    return out;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

Value sigmoid(const Value& a) {
    Value out(sigmoid(a.data()));
    out.node->prev = {a.node};
    out.node->op = "sigmoid";

    Node* o = out.node.get();
    auto an = a.node;

    out.node->backward_fn = [an, o]() {
        an->grad += (o->data * (1 - o->data)) * o->grad;
    };

    return out;
}

Value tanh(const Value& a) {
    Value out(std::tanh(a.data()));
    out.node->prev = {a.node};
    out.node->op = "tanh";

    Node* o = out.node.get();
    auto an = a.node;

    out.node->backward_fn = [an, o]() {
        an->grad += (1 - (o->data * o->data)) * o->grad;
    };

    return out;
}

// ---------------------------------------------------------------------------
// Activation selector
// ---------------------------------------------------------------------------

enum class Activation { Tanh, ReLU, Sigmoid, Linear };

Value activate(const Value& x, Activation act) {
    switch (act) {
        case Activation::Tanh:    return tanh(x);
        case Activation::ReLU:    return relu(x);
        case Activation::Sigmoid: return sigmoid(x);
        case Activation::Linear:  return x;
    }
    return x;
}

// ---------------------------------------------------------------------------
// Neural network
// Note: copying a Neuron / Layer / MLP is shallow. The copy shares the same
// weights (Value is a handle), so training one trains the other.
// ---------------------------------------------------------------------------

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distrib(-1, 1);

class Neuron {
public:
    vector<Value> weights;
    Value bias;
    Activation act;

    explicit Neuron(int nin, Activation activation = Activation::Tanh)
        : bias(distrib(gen)), act(activation) {
        for(int i = 0; i < nin; i++) weights.push_back(distrib(gen));
    }

    Value operator()(const vector<Value>& inputs) {
        if(inputs.size() != weights.size())
            throw invalid_argument("Neuron: expected " + to_string(weights.size())
                                   + " inputs, got " + to_string(inputs.size()));

        Value sum(0.0);
        for(size_t i = 0; i < weights.size(); i++) {
            sum = sum + weights[i] * inputs[i];
        }

        return activate(sum + bias, act);
    }

    vector<Value> parameters() {
        vector<Value> params = weights;
        params.push_back(bias);

        return params;
    }
};

class Layer {
public:
    vector<Neuron> neurons;

    Layer(int nin, int nout, Activation act = Activation::Tanh) {
        for(int i = 0; i < nout; i++) {
            neurons.emplace_back(nin, act);
        }
    }

    vector<Value> operator()(const vector<Value>& inputs) {
        vector<Value> outputs;
        for(auto& neuron : neurons) outputs.push_back(neuron(inputs));
        return outputs;
    }

    vector<Value> parameters() {
        vector<Value> params;
        for(auto& neuron : neurons) {
            for(auto param : neuron.parameters()) {
                params.push_back(param);
            }
        }
        return params;
    }
};

class MLP {
public:
    vector<Layer> layers;

    // Hidden layers use `hidden`; the last layer is always Linear so the
    // output can reach any target (e.g. -1 / +1) whatever the hidden activation.
    MLP(int nin, vector<int> nouts, Activation hidden = Activation::Tanh) {
        vector<int> sizes = {nin};
        for(auto n : nouts) sizes.push_back(n);

        for(size_t i = 0; i < nouts.size(); i++) {
            Activation act = (i + 1 == nouts.size()) ? Activation::Linear : hidden;
            layers.emplace_back(sizes[i], sizes[i + 1], act);
        }
    }

    vector<Value> operator()(const vector<Value>& inputs) {
        vector<Value> activations = inputs;
        for(auto& layer : layers) {
            activations = layer(activations);
        }

        return activations;
    }

    vector<Value> parameters() {
        vector<Value> params;
        for(auto& layer : layers) {
            for(auto& param : layer.parameters()) {
                params.push_back(param);
            }
        }

        return params;
    }
};

// ---------------------------------------------------------------------------
// Self-checks and training demo
// ---------------------------------------------------------------------------

static bool close(double a, double b) { return std::fabs(a - b) < 1e-6; }

void run_self_checks() {
    // Karpathy's single-neuron example: expected values from the Python notebook.
    {
        Value x1(2.0), x2(0.0), w1(-3.0), w2(1.0), b(6.8813735870195432);
        Value n = x1*w1 + x2*w2 + b;
        Value o = tanh(n);
        o.backward();
        assert(close(o.data(), 0.7071067811865476));
        assert(close(x1.grad(), -1.5));
        assert(close(w1.grad(), 1.0));
        assert(close(x2.grad(), 0.5));
        assert(close(w2.grad(), 0.0));
    }

    // Mixed Value / double arithmetic and shared nodes (x*x -> grad 2x).
    {
        Value x(3.0);
        Value y = x * x;
        y.backward();
        assert(close(y.data(), 9.0));
        assert(close(x.grad(), 6.0));

        Value a(2.0), b(4.0);
        Value c = a / b;
        c.backward();
        assert(close(c.data(), 0.5));
        assert(close(a.grad(), 0.25));
        assert(close(b.grad(), -0.125));

        assert(close((a + 2.0).data(), 4.0));
        assert(close((2.0 + a).data(), 4.0));
    }

    // Network shape: (3+1)*4 + (4+1)*4 + (4+1)*1 = 41 parameters.
    {
        MLP m(3, {4, 4, 1});
        assert(m.parameters().size() == 41);
    }

    // Wrong input size must throw, not read out of bounds.
    {
        Neuron n(3);
        bool threw = false;
        try { n(vector<Value>{Value(1.0), Value(2.0)}); }
        catch(const invalid_argument&) { threw = true; }
        assert(threw);
    }

    cout << "self-checks passed\n";
}

void train_demo(Activation hidden, const string& name, int epochs = 200, double lr = 0.05) {
    MLP net(3, {4, 4, 1}, hidden);

    vector<vector<Value>> xs = {
        {Value(2.0), Value(3.0), Value(-1.0)},
        {Value(3.0), Value(-1.0), Value(0.5)},
        {Value(0.5), Value(1.0), Value(1.0)},
        {Value(1.0), Value(1.0), Value(-1.0)},
    };
    vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    auto params = net.parameters();

    cout << "\ntraining with " << name << " hidden layers\n";
    cout << "Epoch\tLoss\n";
    for(int epoch = 1; epoch <= epochs; epoch++) {
        // forward pass: keep everything as Value so the graph stays connected
        Value loss(0.0);
        for(size_t i = 0; i < xs.size(); i++) {
            Value pred = net(xs[i])[0];
            Value diff = pred - Value(ys[i]);
            loss = loss + diff * diff;
        }

        // zero grads before backward, otherwise they accumulate
        for(auto& p : params) p.node->grad = 0.0;
        loss.backward();

        // gradient descent
        for(auto& p : params) p.node->data -= lr * p.node->grad;

        if(epoch == 1 || epoch % 20 == 0) cout << epoch << "\t" << loss.data() << "\n";
    }

    cout << "\nTarget\tPredicted\n";
    for(size_t i = 0; i < xs.size(); i++) {
        cout << ys[i] << "\t" << net(xs[i])[0].data() << "\n";
    }
}

int main(int argc, char** argv) {
    run_self_checks();

    string name = (argc > 1) ? argv[1] : "tanh";
    Activation act = Activation::Tanh;
    if(name == "relu") act = Activation::ReLU;
    else if(name == "sigmoid") act = Activation::Sigmoid;
    else if(name != "tanh") {
        cerr << "usage: " << argv[0] << " [tanh|relu|sigmoid] [seed]\n";
        return 1;
    }

    if(argc > 2) gen.seed(static_cast<unsigned>(stoul(argv[2])));

    train_demo(act, name);
    return 0;
}
