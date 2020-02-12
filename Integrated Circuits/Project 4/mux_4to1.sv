module mux_4to1(
	input logic d3, d2, d1, d0,
	input logic [1:0] sel,
	output logic out
);

assign out = sel[1] ? (sel[0] ? d3 : d2) : (sel[0] ? d1 : d0);
endmodule