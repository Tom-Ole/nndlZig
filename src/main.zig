// Own implementation of a neural network based and slides of my university course "Nerual Networks and Deep Learning".
// Only for leanrning purposes (future Exam).
// Currently only on CPU. Not plans for implenting GPU calculations.

const std = @import("std");
const utils = @import("utils.zig");
const af = @import("activationFunctions.zig");

const ActivationFunction = af.ActivationFunction;

const Layer = struct {
    input_size: usize,
    output_size: usize,
    inputs: []f64,
    output: []f64,

    weights: []f64, // weights table
    biases: []f64,

    pre_output: []f64,
    delta_error: []f64,

    activation_fn: ActivationFunction,

    alloc: *std.mem.Allocator,

    pub fn init(alloc: *std.mem.Allocator, input_size: usize, output_size: usize, activation_fn: ActivationFunction) !Layer {
        std.debug.assert(input_size > 0);
        std.debug.assert(output_size > 0);

        return .{
            .input_size = input_size,
            .output_size = output_size,
            .inputs = try alloc.alloc(f64, input_size),
            .output = try alloc.alloc(f64, output_size),
            .weights = try alloc.alloc(f64, input_size * output_size),
            .pre_output = try alloc.alloc(f64, output_size),
            .delta_error = try alloc.alloc(f64, output_size),
            .biases = try alloc.alloc(f64, output_size),
            .activation_fn = activation_fn,

            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Layer) void {
        self.alloc.free(self.inputs);
        self.alloc.free(self.weights);
        self.alloc.free(self.biases);
        self.alloc.free(self.output);
        self.alloc.free(self.pre_output);
        self.alloc.free(self.delta_error);
    }

    fn set_input(self: *Layer, input: []const f64) void {
        std.debug.assert(self.inputs.len == input.len);

        // assert for finit values to not get any values coming from zero div or overflow.
        for (input) |val| {
            std.debug.assert(std.math.isFinite(val));
        }

        @memcpy(self.inputs, input);
    }

    fn rand_weights(self: *Layer, rng: std.Random) void {
        for (self.weights) |*w| w.* = rng.float(f64) * 0.2 - 0.1;
        for (self.biases) |*b| b.* = 0.0;
    }

    fn compute(self: *Layer) []f64 {
        // assert for finit values to not get any values coming from zero div or overflow.
        for (self.inputs) |val| {
            std.debug.assert(std.math.isFinite(val));
        }

        for (0..self.output_size) |j| {
            var sum: f64 = self.biases[j];
            for (0..self.input_size) |i| {
                sum += self.inputs[i] * self.weights[i * self.output_size + j];
            }
            self.pre_output[j] = sum;
            self.output[j] = self.activation_fn.f(sum);
        }

        return self.output;
    }

    fn validate_gradients(self: *Layer) void {
        // Assert gradients are finite and reasonable
        for (self.delta_error) |delta| {
            std.debug.assert(std.math.isFinite(delta));
            std.debug.assert(@abs(delta) < 1000.0); // Catch exploding gradients
        }
    }

    fn print(self: Layer) void {
        for (self.output) |output| {
            std.debug.print("{d}", .{output});
        }
    }
};

const MLP = struct {
    input_size: usize,
    output_size: usize,
    layer_outputs: []usize,
    layers: []Layer,
    output: []f64,

    ground_truth: []f64,

    alloc: *std.mem.Allocator,

    fn init(alloc: *std.mem.Allocator, input_size: usize, output_size: usize, layer_outputs: []usize, activation_fns: []ActivationFunction) !MLP {

        // Assert valid network architecture
        std.debug.assert(input_size > 0);
        std.debug.assert(output_size > 0);
        std.debug.assert(layer_outputs.len > 0);
        std.debug.assert(activation_fns.len == layer_outputs.len + 1); // +1 for output layer

        const layer_count = layer_outputs.len + 1;

        var layers = try alloc.alloc(Layer, layer_count);

        var prev_size = input_size;

        // hidden layers
        for (0..layer_outputs.len) |i| {
            std.debug.assert(layer_outputs[i] > 0); // at least 1 neuron per layer
            layers[i] = try Layer.init(alloc, prev_size, layer_outputs[i], activation_fns[i]);
            prev_size = layer_outputs[i];
        }

        // output layer
        layers[layer_count - 1] = try Layer.init(alloc, prev_size, output_size, activation_fns[layer_count - 1]);

        return .{
            .alloc = alloc,
            .input_size = input_size,
            .output_size = output_size,
            .layer_outputs = try alloc.dupe(usize, layer_outputs),
            .layers = layers,
            .ground_truth = try alloc.alloc(f64, output_size),
            .output = try alloc.alloc(f64, output_size),
        };
    }

    fn deinit(self: *MLP) void {
        self.alloc.free(self.output);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.alloc.free(self.layers);
    }

    fn rand_weights(self: *MLP) void {
        var prng = std.Random.Xoshiro256.init(2893457);
        const rand = prng.random();
        for (self.layers) |*layer| {
            layer.rand_weights(rand);
        }
    }

    fn compute(self: *MLP, inputs: []const f64) []f64 {
        std.debug.assert(inputs.len == self.input_size);

        // assert for finit values to not get any values coming from zero div or overflow.
        for (inputs) |val| {
            std.debug.assert(std.math.isFinite(val));
        }

        self.layers[0].set_input(inputs);

        for (self.layers, 0..) |*layer, i| {
            const res = layer.compute();
            if (i + 1 < self.layers.len) {
                self.layers[i + 1].set_input(res);
            }
        }
        @memcpy(self.output, self.layers[self.layers.len - 1].output);

        // assert for finit values to not get any values coming from zero div or overflow.
        for (self.output) |val| {
            std.debug.assert(std.math.isFinite(val));
        }

        return self.output;
    }

    fn backprog(self: *MLP, eta: f64) void {

        // assert for finit values to not get any values coming from zero div or overflow.
        for (self.ground_truth) |val| {
            std.debug.assert(std.math.isFinite(val));
        }

        const last_layer = &self.layers[self.layers.len - 1];
        for (0..self.output_size) |j| {
            const error_val = self.output[j] - self.ground_truth[j];
            std.debug.assert(std.math.isFinite(error_val));

            const derivative = last_layer.activation_fn.df(last_layer.pre_output[j]);
            std.debug.assert(std.math.isFinite(derivative));

            last_layer.delta_error[j] = error_val * derivative;
        }

        last_layer.validate_gradients();

        // Output already calculatet;
        // Input doesnt need to be affected;
        var k: usize = self.layers.len - 2; // start at second-to-last
        while (true) : (k -= 1) {
            const current = &self.layers[k];
            const next = &self.layers[k + 1];

            for (0..current.output_size) |current_idx| {
                var sum: f64 = 0;
                for (0..next.output_size) |next_idx| {
                    sum += next.delta_error[next_idx] * next.weights[current_idx * next.output_size + next_idx];
                }

                std.debug.assert(std.math.isFinite(sum));

                const derivative = current.activation_fn.df(current.pre_output[current_idx]);
                std.debug.assert(std.math.isFinite(derivative));

                current.delta_error[current_idx] = sum * derivative;
            }

            current.validate_gradients();

            if (k == 0) break;
        }

        for (self.layers) |*layer| {
            for (0..layer.output_size) |j| {
                for (0..layer.input_size) |i| {
                    const grad = layer.inputs[i] * layer.delta_error[j];
                    std.debug.assert(std.math.isFinite(grad));

                    layer.weights[i * layer.output_size + j] -= eta * grad;
                    std.debug.assert(std.math.isFinite(layer.weights[i * layer.output_size + j]));
                }
                layer.biases[j] -= eta * layer.delta_error[j];
                std.debug.assert(std.math.isFinite(layer.biases[j]));
            }
        }
    }

    fn set_ground_truth(self: *MLP, target: []const f64) void {
        std.debug.assert(target.len == self.output_size);
        for (target) |val| {
            std.debug.assert(std.math.isFinite(val));
        }
        @memcpy(self.ground_truth, target);
    }

    fn print(self: MLP) void {
        for (self.output) |output| {
            std.debug.print("{d}", .{output});
        }
    }
};

fn mse(pred: []const f64, target: []const f64) f64 {
    var sum: f64 = 0;
    for (pred, 0..) |p, i| {
        const diff = p - target[i];
        sum += diff * diff;
    }
    return sum / @as(f64, @floatFromInt(pred.len));
}

fn cross_entropy_loss(pred: []const f64, target: []const f64) f64 {
    std.debug.assert(pred.len == target.len);

    var loss: f64 = 0.0;
    const epsilon: f64 = 1e-15;

    for (pred, 0..) |p, i| {
        // Clamp predictions to avoid numerical issues
        const clamped_p = @max(epsilon, @min(1.0 - epsilon, p));
        loss -= target[i] * std.math.log(f64, 2, clamped_p);
    }

    return loss;
}

fn print_list_float(comptime T: type, arr: []const T) void {
    for (arr) |elem| {
        std.debug.print("{d:.3}, ", .{elem});
    }
}

fn max_idx_list_float(comptime T: type, arr: []const T) usize {
    var max_idx: usize = 0;

    for (0..arr.len) |idx| {
        if (arr[idx] > arr[max_idx]) max_idx = idx;
    }

    return max_idx;
}

pub fn main() !void {
    const TrackingAllocator = utils.TrackingAllocator;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var tracking_allocator = TrackingAllocator.init(gpa.allocator());
    var alloc = tracking_allocator.allocator();

    var profiler = utils.Profiler.init();

    var hidden_layers = [_]usize{ 6, 3, 6, 9 };
    var activations_fns = [_]ActivationFunction{ af.LEAKY_RELU, af.TANH, af.SWISH, af.TANH, af.SIGMOID };

    // Architecture: 2 inputs → [2 hidden] → 1 output
    var mlp = try MLP.init(&alloc, 9, 9, hidden_layers[0..], activations_fns[0..]);
    // defer mlp.deinit();

    mlp.rand_weights();

    // Biggest number dataset
    const inputs = [_][9]f64{
        .{ 1, 2, 3, 4, 5, 6, 7, 8, 9 },
        .{ -1, -2, -3, -4, -5, -6, -7, -9, -9 },
        .{ 12, 85, 42, 6, 87, 234, 1, 8, 9 },
        .{ 0, -2, 56, -568, 99, 1, 0, 25, -212 },
    };
    const targets = [_][9]f64{
        .{ 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        .{ 1, 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 0, 0, 0, 0, 0, 1, 0, 0, 0 },
        .{ 0, 0, 0, 0, 1, 0, 0, 0, 0 },
    };

    const epochs = 100_000;
    const eta = 0.1; // learning rate

    // Train
    for (0..epochs) |epoch| {
        var epoch_loss_mse: f64 = 0;
        var epoch_loss_ce: f64 = 0;

        for (inputs, 0..) |inp, idx| {
            const out = mlp.compute(inp[0..]); // forward pass
            mlp.set_ground_truth(&targets[idx]);
            mlp.backprog(eta);
            epoch_loss_mse += mse(out, &targets[idx]);
            epoch_loss_ce += cross_entropy_loss(out, &targets[idx]);
        }

        if (epoch % 10_000 == 0) {
            std.debug.print("Epoch {d}, mse={d:.5}, cross_entropy={d:.5}\n", .{ epoch, epoch_loss_mse, epoch_loss_ce });
        }
    }

    std.debug.print("\n--- Output ---\n", .{});
    for (inputs, 0..) |inp, idx| {
        const out = mlp.compute(inp[0..]);
        std.debug.print("Target: ", .{});
        print_list_float(f64, targets[idx][0..]);
        std.debug.print("\n", .{});
        std.debug.print("Output: ", .{});
        print_list_float(f64, out);
        std.debug.print("\n", .{});

        const max_out_idx = max_idx_list_float(f64, out);
        const max_target_idx = max_idx_list_float(f64, targets[idx][0..]);
        std.debug.print("max_out_idx: {} \nmax_target_idx: {} \n \n", .{ max_out_idx, max_target_idx });
    }

    const test_inputs = [_][9]f64{
        .{ 54, 12, 6, -1, 5, 5, 87, 23, 78 },
        .{ -1, 31, -1235, 14636, 52, 74, 1, 0, 7 },
    };
    const test_targets = [_][9]f64{
        .{ 0, 0, 0, 0, 0, 0, 1, 0, 0 },
        .{ 0, 0, 0, 1, 0, 0, 0, 0, 0 },
    };

    std.debug.print("\n--- Test Output ---\n", .{});
    for (test_inputs, 0..) |inp, idx| {
        const out = mlp.compute(inp[0..]);
        std.debug.print("Target: ", .{});
        print_list_float(f64, test_targets[idx][0..]);
        std.debug.print("\n", .{});
        std.debug.print("Output: ", .{});
        print_list_float(f64, out);
        std.debug.print("\n", .{});

        const max_out_idx = max_idx_list_float(f64, out);
        const max_target_idx = max_idx_list_float(f64, test_targets[idx][0..]);
        std.debug.print("max_out_idx: {} \nmax_target_idx: {} \n \n", .{ max_out_idx, max_target_idx });
    }

    std.debug.print("\n--- Profiling ---\n", .{});
    std.debug.print("Total Time:                     {d:.3}sec \n", .{profiler.getTotalTime()});
    std.debug.print("Peak Memory:                    {d:.3}KB \n", .{tracking_allocator.getPeakMemoryKB()});
    std.debug.print("Current Memory (before deinit): {d:.3}KB \n", .{tracking_allocator.getCurrentMemoryKB()});

    mlp.deinit();
    std.debug.print("Current Memory (after deinit):  {d:.3}KB \n", .{tracking_allocator.getCurrentMemoryKB()});
}

//
//
// Tests
//
//

test "MLP XOR" {
    const TrackingAllocator = utils.TrackingAllocator;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var tracking_allocator = TrackingAllocator.init(gpa.allocator());
    var alloc = tracking_allocator.allocator();

    // var profiler = utils.Profiler.init();

    var hidden_layers = [_]usize{2};
    var activations_fns = [_]ActivationFunction{ af.SIGMOID, af.SIGMOID };

    // Architecture: 2 inputs → [2 hidden] → 1 output
    var mlp = try MLP.init(&alloc, 2, 1, hidden_layers[0..], activations_fns[0..]);
    defer mlp.deinit();

    mlp.rand_weights();

    // XOR dataset
    const inputs = [_][2]f64{
        .{ 0, 0 },
        .{ 0, 1 },
        .{ 1, 0 },
        .{ 1, 1 },
    };
    const targets = [_][1]f64{
        .{0},
        .{1},
        .{1},
        .{0},
    };

    const epochs = 10_000;
    const eta = 0.3; // learning rate

    // Train
    for (0..epochs) |_| {
        var epoch_loss: f64 = 0;

        for (inputs, 0..) |inp, idx| {
            const out = mlp.compute(inp[0..]); // forward pass
            mlp.set_ground_truth(&targets[idx]);
            mlp.backprog(eta);
            epoch_loss += mse(out, &targets[idx]);
        }
    }

    std.debug.print("\n--- Output ---\n", .{});

    for (inputs, 0..) |inp, idx| {
        const out = mlp.compute(inp[0..]);
        const out_bin: f64 = if (out[0] > 0.95) 1 else 0;
        try std.testing.expect(out_bin == targets[idx][0]);
        std.debug.print("in={any} -> out={d} (raw {d:.3}, target {d})\n", .{ inp, out_bin, out[0], targets[idx][0] });
    }
}
