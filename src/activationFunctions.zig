const std = @import("std");

pub const ActivationFunction = struct {
    f: *const fn (f64) f64,
    df: *const fn (f64) f64,
};

fn sigmoid(t: f64) f64 {
    return 0.5 * (1 + std.math.tanh(0.5 * t));
}
fn d_sigmoid(t: f64) f64 {
    const s = sigmoid(t);
    return s * (1 - s);
}

pub const SIGMOID: ActivationFunction = .{
    .f = sigmoid,
    .df = d_sigmoid,
};

fn linear(x: f64) f64 {
    return x;
}

fn d_linear(_: f64) f64 {
    return 1.0;
}

pub const LINEAR: ActivationFunction = .{
    .f = linear,
    .df = d_linear,
};

fn relu(x: f64) f64 {
    return @max(0.0, x);
}

fn d_relu(x: f64) f64 {
    if (x < 0) return 0;
    return 1;
}

pub const RELU: ActivationFunction = .{
    .f = relu,
    .df = d_relu,
};

fn softmax_single(x: f64) f64 {
    return std.math.exp(x);
}

fn d_softmax_single(x: f64) f64 {
    const s = std.math.exp(x);
    return s * (1.0 - s);
}

pub const SOFTMAX_SINGLE: ActivationFunction = .{
    .f = softmax_single,
    .df = d_softmax_single,
};

fn tahn(x: f64) f64 {
    return std.math.tanh(x);
}

fn d_tanh(x: f64) f64 {
    const t = std.math.tanh(x);
    return 1.0 - t * t;
}

pub const TANH: ActivationFunction = .{
    .f = tahn,
    .df = d_tanh,
};

fn leaky_relu(x: f64) f64 {
    return if (x > 0) x else 0.01 * x;
}

fn d_leaky_relu(x: f64) f64 {
    return if (x > 0) 1.0 else 0.01;
}

pub const LEAKY_RELU: ActivationFunction = .{
    .f = leaky_relu,
    .df = d_leaky_relu,
};

fn swish(x: f64) f64 {
    return x / (1.0 + std.math.exp(-x));
}

fn d_swish(x: f64) f64 {
    const exp_neg_x = std.math.exp(-x);
    const sigmoid_x = 1.0 / (1.0 + exp_neg_x);
    return sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x);
}

pub const SWISH: ActivationFunction = .{
    .f = swish,
    .df = d_swish,
};
