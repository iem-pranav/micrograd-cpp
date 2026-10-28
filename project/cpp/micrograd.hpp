// micrograd in C++ (header-only)
// A port of Andrej Karpathy's micrograd (scalar autograd engine + tiny MLP).
// Derived from the author's micrograd.cpp; behaves like reference/micrograd_extended.cpp,
// except that sigmoid uses an overflow-safe form (see below).
//
// Header-only: every function and global is `inline`, so including this header
// from several .cpp files does not break the "one definition rule" at link time.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Autograd engine
// ---------------------------------------------------------------------------

struct Node {
    double data;
    double grad = 0.0;
    std::vector<std::shared_ptr<Node>> prev;   // children (the inputs that produced this node)
    std::string op;
    std::function<void()> backward_fn = [](){};  // like Python's `lambda: None`

    Node(double d, std::vector<std::shared_ptr<Node>> children = {}, std::string o = "")
        : data(d), prev(std::move(children)), op(std::move(o)) {}
};

// Recursive: the depth grows with the number of data points (the loss is a long
// chain of `+`). Do not raise the point limit without making this iterative.
inline void build_topo(Node* v, std::unordered_set<Node*>& visited, std::vector<Node*>& topo) {
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
    std::shared_ptr<Node> node;

    Value(double d = 0.0) {
        node = std::make_shared<Node>(d);
    }

    double data() const { return node->data; }
    double grad() const { return node->grad; }

    void backward() {
        std::unordered_set<Node*> visited;
        std::vector<Node*> topo;
        node->grad = 1.0;
        build_topo(node.get(), visited, topo);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) (*it)->backward_fn();
    }
};

// Each op's closure keeps its children alive (an, bn) but only holds a raw
// pointer (o) to its own node: otherwise node -> closure -> node would be a
// shared_ptr cycle and nothing would ever be freed.

inline Value operator+(const Value& a, const Value& b) {
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

inline Value operator-(const Value& a, const Value& b) {
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

inline Value operator-(const Value& a) {
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

inline Value operator*(const Value& a, const Value& b) {
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

inline Value pow(const Value& a, double p) {
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

inline Value operator/(const Value& a, const Value& b) {
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

inline Value exp(const Value& a) {
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

inline Value relu(const Value& a) {
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

// Overflow-safe sigmoid. The plain 1/(1+exp(-x)) computes exp(1000) for x = -1000;
// that is inf in C++ but an OverflowError in Python. Using exp of a value <= 0 on
// both branches avoids that. The Python port uses exactly the same formula.
inline double sigmoid(double x) {
    if(x >= 0) return 1.0 / (1.0 + std::exp(-x));
    double e = std::exp(x);
    return e / (1.0 + e);
}

inline Value sigmoid(const Value& a) {
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

inline Value tanh(const Value& a) {
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

inline Value activate(const Value& x, Activation act) {
    switch (act) {
        case Activation::Tanh:    return tanh(x);
        case Activation::ReLU:    return relu(x);
        case Activation::Sigmoid: return sigmoid(x);
        case Activation::Linear:  return x;
    }
    return x;
}

// ---------------------------------------------------------------------------
// Random numbers
// A fixed default seed (not std::random_device) so runs are reproducible.
// `inline` variables (C++17) give one shared RNG even though this is a header.
// ---------------------------------------------------------------------------

inline std::mt19937 gen(1);
inline std::uniform_real_distribution<double> distrib(-1, 1);

inline void seed_rng(unsigned seed) {
    gen.seed(seed);
    distrib.reset();   // drop any state cached by the distribution
}

// ---------------------------------------------------------------------------
// Neural network
// Note: copying a Neuron / Layer / MLP is shallow. The copy shares the same
// weights (Value is a handle), so training one trains the other.
// ---------------------------------------------------------------------------

class Neuron {
public:
    std::vector<Value> weights;
    Value bias;
    Activation act;

    explicit Neuron(int nin, Activation activation = Activation::Tanh)
        : bias(distrib(gen)), act(activation) {
        for(int i = 0; i < nin; i++) weights.push_back(distrib(gen));
    }

    Value operator()(const std::vector<Value>& inputs) {
        if(inputs.size() != weights.size())
            throw std::invalid_argument("Neuron: expected " + std::to_string(weights.size())
                                        + " inputs, got " + std::to_string(inputs.size()));

        Value sum(0.0);
        for(size_t i = 0; i < weights.size(); i++) {
            sum = sum + weights[i] * inputs[i];
        }

        return activate(sum + bias, act);
    }

    std::vector<Value> parameters() {
        std::vector<Value> params = weights;
        params.push_back(bias);

        return params;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int nin, int nout, Activation act = Activation::Tanh) {
        for(int i = 0; i < nout; i++) {
            neurons.emplace_back(nin, act);
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& inputs) {
        std::vector<Value> outputs;
        for(auto& neuron : neurons) outputs.push_back(neuron(inputs));
        return outputs;
    }

    std::vector<Value> parameters() {
        std::vector<Value> params;
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
    std::vector<Layer> layers;

    // Hidden layers use `hidden`; the last layer is always Linear so the
    // output can reach any target (e.g. -1 / +1) whatever the hidden activation.
    MLP(int nin, std::vector<int> nouts, Activation hidden = Activation::Tanh) {
        std::vector<int> sizes = {nin};
        for(auto n : nouts) sizes.push_back(n);

        for(size_t i = 0; i < nouts.size(); i++) {
            Activation act = (i + 1 == nouts.size()) ? Activation::Linear : hidden;
            layers.emplace_back(sizes[i], sizes[i + 1], act);
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& inputs) {
        std::vector<Value> activations = inputs;
        for(auto& layer : layers) {
            activations = layer(activations);
        }

        return activations;
    }

    // Order: for each layer, for each neuron, its weights in input order, then its bias.
    std::vector<Value> parameters() {
        std::vector<Value> params;
        for(auto& layer : layers) {
            for(auto& param : layer.parameters()) {
                params.push_back(param);
            }
        }

        return params;
    }

    // Overwrite every parameter, in parameters() order (used for `init_weights`).
    void set_parameters(const std::vector<double>& values) {
        std::vector<Value> params = parameters();
        if(values.size() != params.size())
            throw std::invalid_argument("expected " + std::to_string(params.size())
                                        + " weights, got " + std::to_string(values.size()));
        for(size_t i = 0; i < params.size(); i++) params[i].node->data = values[i];
    }
};
