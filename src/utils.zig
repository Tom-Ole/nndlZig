const std = @import("std");

pub const Profiler = struct {
    start_time: i128,

    pub fn init() Profiler {
        return Profiler{
            .start_time = std.time.nanoTimestamp(),
        };
    }

    pub fn getTotalTime(self: *Profiler) f64 {
        return @as(f64, @floatFromInt(std.time.nanoTimestamp() - self.start_time)) / 1e9;
    }
};

// Tracker Allocator for Testing and debuggin.
// Created with the help of LLMs. I am just started to learn Zig and the concept of allocators.
// (Will revisted later to optimize and fully understand it) [Maybe a full tracking libary]
pub const TrackingAllocator = struct {
    child: std.mem.Allocator,
    total_allocated: usize,
    peak_allocated: usize,
    current_allocated: usize,

    pub fn init(child: std.mem.Allocator) TrackingAllocator {
        return TrackingAllocator{
            .child = child,
            .total_allocated = 0,
            .peak_allocated = 0,
            .current_allocated = 0,
        };
    }

    pub fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return std.mem.Allocator{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
                .remap = remap,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.child.rawAlloc(len, ptr_align, ret_addr);
        if (result) |_| {
            self.total_allocated += len;
            self.current_allocated += len;
            if (self.current_allocated > self.peak_allocated) {
                self.peak_allocated = self.current_allocated;
            }
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.child.rawResize(buf, buf_align, new_len, ret_addr)) {
            if (new_len > buf.len) {
                const diff = new_len - buf.len;
                self.total_allocated += diff;
                self.current_allocated += diff;
                if (self.current_allocated > self.peak_allocated) {
                    self.peak_allocated = self.current_allocated;
                }
            } else {
                const diff = buf.len - new_len;
                self.current_allocated -= diff;
            }
            return true;
        }
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.child.rawFree(buf, buf_align, ret_addr);
        self.current_allocated -= buf.len;
    }

    fn remap(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));

        // Try to remap using the child allocator
        if (self.child.rawResize(buf, buf_align, new_len, ret_addr)) {
            // Successful resize - update our tracking
            if (new_len > buf.len) {
                const diff = new_len - buf.len;
                self.total_allocated += diff;
                self.current_allocated += diff;
                if (self.current_allocated > self.peak_allocated) {
                    self.peak_allocated = self.current_allocated;
                }
            } else {
                const diff = buf.len - new_len;
                self.current_allocated -= diff;
            }
            return buf.ptr;
        } else {
            // Resize failed, try allocate + copy + free
            const new_ptr = self.child.rawAlloc(new_len, buf_align, ret_addr) orelse return null;

            // Update tracking for new allocation
            self.total_allocated += new_len;
            self.current_allocated += new_len;
            if (self.current_allocated > self.peak_allocated) {
                self.peak_allocated = self.current_allocated;
            }

            // Copy data
            const copy_len = @min(buf.len, new_len);
            @memcpy(new_ptr[0..copy_len], buf[0..copy_len]);

            // Free old memory and update tracking
            self.child.rawFree(buf, buf_align, ret_addr);
            self.current_allocated -= buf.len;

            return new_ptr;
        }
    }

    pub fn getPeakMemoryMB(self: *TrackingAllocator) f64 {
        return @as(f64, @floatFromInt(self.peak_allocated)) / 1024.0 / 1024.0;
    }

    pub fn getPeakMemoryKB(self: *TrackingAllocator) f64 {
        return @as(f64, @floatFromInt(self.peak_allocated)) / 1024.0;
    }

    pub fn getPeakMemoryBytes(self: *TrackingAllocator) usize {
        return self.peak_allocated;
    }

    pub fn getCurrentMemoryMB(self: *TrackingAllocator) f64 {
        return @as(f64, @floatFromInt(self.current_allocated)) / 1024.0 / 1024.0;
    }

    pub fn getCurrentMemoryKB(self: *TrackingAllocator) f64 {
        return @as(f64, @floatFromInt(self.current_allocated)) / 1024.0;
    }

    pub fn getCurrentMemoryBytes(self: *TrackingAllocator) usize {
        return self.current_allocated;
    }

    pub fn getTotalAllocated(self: *TrackingAllocator) usize {
        return self.total_allocated;
    }
};
