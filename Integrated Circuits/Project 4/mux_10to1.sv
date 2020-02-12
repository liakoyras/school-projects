module mux_10to1(
	input logic d9, d8, d7, d6, d5, d4, d3, d2, d1, d0,
	input logic [3:0] select,
	output logic X
);

logic out_internal [2:0];
logic zero = 1'b0;

mux_4to1 m1 (
	.d3(d3), .d2(d2), d1(d1), d0(d0),
	.sel[1:0](select[1:0]),
	.out(out_internal[0])
);

mux_4to1 m2 (
	.d3(d7), .d2(d6), d1(d5), d0(d4),
	.sel[1:0](select[1:0]),
	.out(out_internal[1])
);

mux_4to1 m3 (
	.d3(zero), .d2(zero), d1(d9), d0(d8),
	.sel[1:0](select[1:0]),
	.out(out_internal[2])
);

mux_4to1 m4 (
	.d3(zero), .d2(out_internal[2]), d1(out_internal[1]), d0(out_internal[0]),
	.sel[1:0](select[3:2]),
	.out(X)
);
endmodule