// C++ self-checks for cpp/micrograd.hpp.
// Run by `make test` (-O2) and `make asan` (sanitizers: also catches leaks in the graph).
// Prints every failed check and exits non-zero if any failed.

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cpp/micrograd.hpp"

using namespace std;

int failures = 0;

void check(bool ok, const string& what) {
    if(!ok) {
        cout << "FAIL: " << what << "\n";
        failures++;
    }
}

bool close(double a, double b, double tol = 1e-6) { return std::fabs(a - b) < tol; }

// Compare the gradient from backward() with a central finite difference.
void check_unary_grad(const string& name, function<Value(const Value&)> f, double x0) {
    Value x(x0);
    Value y = f(x);
    y.backward();
    double h = 1e-6;
    double numeric = (f(Value(x0 + h)).data() - f(Value(x0 - h)).data()) / (2 * h);
    check(close(x.grad(), numeric, 1e-5), name + " grad at " + to_string(x0));
}

void check_binary_grad(const string& name, function<Value(const Value&, const Value&)> f,
                       double a0, double b0) {
    Value a(a0), b(b0);
    Value y = f(a, b);
    y.backward();
    double h = 1e-6;
    double da = (f(Value(a0 + h), Value(b0)).data() - f(Value(a0 - h), Value(b0)).data()) / (2 * h);
    double db = (f(Value(a0), Value(b0 + h)).data() - f(Value(a0), Value(b0 - h)).data()) / (2 * h);
    check(close(a.grad(), da, 1e-5), name + " grad wrt a");
    check(close(b.grad(), db, 1e-5), name + " grad wrt b");
}

int main() {
    // Karpathy's single-neuron example: expected values from the Python notebook.
    {
        Value x1(2.0), x2(0.0), w1(-3.0), w2(1.0), b(6.8813735870195432);
        Value n = x1*w1 + x2*w2 + b;
        Value o = tanh(n);
        o.backward();
        check(close(o.data(), 0.7071067811865476), "notebook neuron output 0.7071");
        check(close(x1.grad(), -1.5), "notebook x1.grad -1.5");
        check(close(w1.grad(), 1.0), "notebook w1.grad 1");
        check(close(x2.grad(), 0.5), "notebook x2.grad 0.5");
        check(close(w2.grad(), 0.0), "notebook w2.grad 0");
    }

    // Mixed Value / double arithmetic and shared nodes (x*x -> grad 2x).
    {
        Value x(3.0);
        Value y = x * x;
        y.backward();
        check(close(y.data(), 9.0) && close(x.grad(), 6.0), "x*x and grad 2x");

        Value a(2.0), b(4.0);
        Value c = a / b;
        c.backward();
        check(close(c.data(), 0.5), "a/b value");
        check(close(a.grad(), 0.25) && close(b.grad(), -0.125), "a/b grads");

        check(close((a + 2.0).data(), 4.0) && close((2.0 + a).data(), 4.0), "Value + double both ways");
    }

    // Every op's gradient against finite differences.
    {
        check_binary_grad("+", [](const Value& a, const Value& b) { return a + b; }, 1.5, -0.7);
        check_binary_grad("-", [](const Value& a, const Value& b) { return a - b; }, 1.5, -0.7);
        check_binary_grad("*", [](const Value& a, const Value& b) { return a * b; }, 1.5, -0.7);
        check_binary_grad("/", [](const Value& a, const Value& b) { return a / b; }, 1.5, -0.7);
        check_unary_grad("neg", [](const Value& a) { return -a; }, 0.8);
        check_unary_grad("pow 3", [](const Value& a) { return pow(a, 3.0); }, 0.8);
        check_unary_grad("exp", [](const Value& a) { return exp(a); }, 0.8);
        check_unary_grad("relu +", [](const Value& a) { return relu(a); }, 0.8);
        check_unary_grad("relu -", [](const Value& a) { return relu(a); }, -0.8);
        check_unary_grad("tanh", [](const Value& a) { return tanh(a); }, 0.8);
        check_unary_grad("sigmoid +", [](const Value& a) { return sigmoid(a); }, 0.8);
        check_unary_grad("sigmoid -", [](const Value& a) { return sigmoid(a); }, -0.8);
    }

    // Sigmoid must not overflow for large |x|.
    {
        check(close(sigmoid(0.0), 0.5), "sigmoid(0) = 0.5");
        check(sigmoid(1000.0) == 1.0, "sigmoid(1000) = 1");
        check(sigmoid(-1000.0) == 0.0, "sigmoid(-1000) = 0, not NaN");
        Value x(-1000.0);
        Value y = sigmoid(x);
        y.backward();
        check(std::isfinite(x.grad()), "sigmoid grad finite at -1000");
    }

    // Activation selector, and the last layer is always linear.
    {
        check(close(activate(Value(-2.0), Activation::Linear).data(), -2.0), "activate Linear");
        check(close(activate(Value(-2.0), Activation::ReLU).data(), 0.0), "activate ReLU");
        MLP m(2, {3, 3, 1}, Activation::ReLU);
        check(m.layers[0].neurons[0].act == Activation::ReLU, "hidden layer uses ReLU");
        check(m.layers[1].neurons[0].act == Activation::ReLU, "second hidden layer uses ReLU");
        check(m.layers[2].neurons[0].act == Activation::Linear, "last layer is Linear");
    }

    // Network shape: (3+1)*4 + (4+1)*4 + (4+1)*1 = 41 parameters.
    {
        MLP m(3, {4, 4, 1});
        check(m.parameters().size() == 41, "MLP(3, {4,4,1}) has 41 parameters");
    }

    // Parameter order: per layer, per neuron, weights in input order, then bias.
    {
        MLP m(2, {3, 1});   // (2+1)*3 + (3+1)*1 = 13 parameters
        vector<double> values;
        for(int i = 0; i < 13; i++) values.push_back(i);
        m.set_parameters(values);
        check(m.layers[0].neurons[0].weights[0].data() == 0, "order: layer0 neuron0 w0");
        check(m.layers[0].neurons[0].weights[1].data() == 1, "order: layer0 neuron0 w1");
        check(m.layers[0].neurons[0].bias.data() == 2, "order: layer0 neuron0 bias");
        check(m.layers[0].neurons[1].weights[0].data() == 3, "order: layer0 neuron1 w0");
        check(m.layers[1].neurons[0].bias.data() == 12, "order: last bias is last");
        auto params = m.parameters();
        bool in_order = true;
        for(size_t i = 0; i < params.size(); i++) if(params[i].data() != i) in_order = false;
        check(in_order, "parameters() returns set_parameters() order");

        bool threw = false;
        try { m.set_parameters({1.0, 2.0}); }
        catch(const invalid_argument&) { threw = true; }
        check(threw, "set_parameters with wrong length throws");
    }

    // Wrong input size must throw, not read out of bounds.
    {
        Neuron n(3);
        bool threw = false;
        try { n(vector<Value>{Value(1.0), Value(2.0)}); }
        catch(const invalid_argument&) { threw = true; }
        check(threw, "Neuron with wrong input size throws");
    }

    // Seeded RNG: same seed -> same weights, different seed -> different weights.
    {
        seed_rng(7);
        MLP a(2, {4, 1});
        seed_rng(7);
        MLP b(2, {4, 1});
        seed_rng(8);
        MLP c(2, {4, 1});
        auto pa = a.parameters(), pb = b.parameters(), pc = c.parameters();
        bool same = true, differs = false;
        for(size_t i = 0; i < pa.size(); i++) {
            if(pa[i].data() != pb[i].data()) same = false;
            if(pa[i].data() != pc[i].data()) differs = true;
        }
        check(same, "same seed gives same weights");
        check(differs, "different seed gives different weights");
    }

    // Copies are shallow: they share weights.
    {
        MLP a(2, {2, 1});
        MLP b = a;
        a.layers[0].neurons[0].weights[0].node->data = 42.0;
        check(b.layers[0].neurons[0].weights[0].data() == 42.0, "MLP copy shares weights");
    }

    // A small training run must lower the loss.
    {
        seed_rng(1);
        MLP net(2, {4, 1});
        vector<vector<Value>> xs = {{Value(1.0), Value(1.0)}, {Value(-1.0), Value(-1.0)},
                                    {Value(1.0), Value(-1.0)}, {Value(-1.0), Value(1.0)}};
        vector<double> ys = {1.0, 1.0, -1.0, -1.0};
        auto params = net.parameters();
        double first = 0.0, last = 0.0;
        for(int epoch = 1; epoch <= 300; epoch++) {
            Value loss(0.0);
            for(size_t i = 0; i < xs.size(); i++) {
                Value diff = net(xs[i])[0] - Value(ys[i]);
                loss = loss + diff * diff;
            }
            for(auto& p : params) p.node->grad = 0.0;
            loss.backward();
            for(auto& p : params) p.node->data -= 0.05 * p.node->grad;
            if(epoch == 1) first = loss.data();
            last = loss.data();
        }
        check(last < first * 0.5, "training on XOR lowers the loss");
    }

    if(failures > 0) {
        cout << failures << " self-check(s) failed\n";
        return 1;
    }
    cout << "self-checks passed\n";
    return 0;
}
