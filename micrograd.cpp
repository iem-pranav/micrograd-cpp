#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <unordered_set>
#include <random>
using namespace std;

struct Node {
    double data;
    double grad = 0.0;
    vector<shared_ptr<Node>> prev;   // children
    string op;
    function<void()> backward_fn = [](){};  // like Python's lambda: None

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

class Value {
public:
    shared_ptr<Node> node;

    Value(double d = 0.0) /* TODO: make the node */ {
        node = make_shared<Node>(d);
    }

    // helper so you can write v.data() / v.grad()
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

Value operator+(const Value& a, const Value& b) {
    Value out(a.data() + b.data());
    out.node->prev = {a.node, b.node};
    out.node->op = "+";

    Node* o = out.node.get();   // raw pointer to out's node
    auto an = a.node;           // copies of the shared_ptrs
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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distrib(-1, 1);

class Neuron {
    
public:
    
    vector<Value> weights;
    Value bias;

    Neuron(int nin) {
        for(int i = 0; i < nin; i++) weights.push_back(distrib(gen));
        bias = distrib(gen);
    }

    Value operator()(const vector<Value>& inputs) {
        Value sum(0.0);
        for(size_t i = 0; i < weights.size(); i++) {
            sum = sum + weights[i] * inputs[i];
        }

        Value act = sum + bias;
        Value out = tanh(act);

        return out;
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

    Layer(int nin, int nout) {
        for(int i = 0; i < nout; i++) {
            neurons.emplace_back(nin);
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
    
    MLP(int nin, vector<int> nouts) {
        vector<int> sizes = {nin};
        for(auto n : nouts) sizes.push_back(n);

        for(size_t i = 0; i < nouts.size(); i++) {
            layers.emplace_back(sizes[i], sizes[i + 1]);
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

int main() {

    {
        Neuron n(3);
        vector<Value> x = {Value(2.0), Value(3.0), Value(-1.0)};
        Value out = n(x);
        out.backward();

        // cout << out.data() << endl;
        // cout << n.parameters().size() << endl;
        // for(auto param : n.parameters()) cout << param.grad() << endl;
    }

    {
        Layer l(3, 4);
        vector<Value> x = {Value(2.0), Value(3.0), Value(-1.0)};
        auto out = l(x);
        out[0].backward();

        // cout << out.size() << endl;                 // expect 4
        // cout << l.parameters().size() << endl;      // expect 16
        // for(auto param : l.parameters()) cout << param.grad() << endl;
    }

    {
        MLP m(3, {4, 4, 1});
        vector<Value> x = {Value(2.0), Value(3.0), Value(-1.0)};
        auto out = m(x);
        out[0].backward();
        cout << out.size() << endl;                 // expect 1
        cout << m.parameters().size() << endl;      // expect 41
        for(auto param : m.parameters()) cout << param.grad() << "\n";
    }
{
    MLP net(3, {4, 4, 1});

    vector<vector<Value>> xs = {
        {Value(2.0), Value(3.0), Value(-1.0)},
        {Value(3.0), Value(-1.0), Value(0.5)},
        {Value(0.5), Value(1.0), Value(1.0)},
        {Value(1.0), Value(1.0), Value(-1.0)},
    };
    vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    auto params = net.parameters();
    double lr = 0.05;

    cout << "Epoch\tLoss" << endl;
    for(int epoch = 1; epoch <= 1000; epoch++) {

        // forward pass: loss is a Value, so the graph stays connected
        Value loss(0.0);
        for(size_t i = 0; i < xs.size(); i++) {
            Value pred = net(xs[i])[0];
            Value diff = pred - Value(ys[i]);
            loss = loss + diff * diff;
        }

        // zero grads BEFORE backward, otherwise they accumulate
        for(auto& p : params) p.node->grad = 0.0;

        loss.backward();

        // gradient descent: step against the gradient
        for(auto& p : params) p.node->data -= lr * p.node->grad;

        if(epoch % 10 == 0 || epoch == 1) cout << epoch << "\t" << loss.data() << endl;
    }

    cout << "\nTarget\tPredicted" << endl;
    for(size_t i = 0; i < xs.size(); i++) {
        cout << ys[i] << "\t" << net(xs[i])[0].data() << endl;
    }
}


    return 0;
}