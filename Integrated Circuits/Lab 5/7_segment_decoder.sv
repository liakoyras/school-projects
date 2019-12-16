module 7_segment_decoder(
	input logic [2:0] in_3_bits,
	output logic [6:0] out_7_bits
)
always_comb begin
	case (in_3_bits)
		3'b000: out_7_bits = 7'b0111111;
		3'b001: out_7_bits = 7'b0000110;
		3'b010: out_7_bits = 7'b1011011;
		3'b011: out_7_bits = 7'b1001111;
		3'b100: out_7_bits = 7'b1100110;
		3'b101: out_7_bits = 7'b1101101;
		3'b110: out_7_bits = 7'b1111101;
		3'b111: out_7_bits = 7'b0000111;
		default: out_7_bits = 7'b0000000;
	endcase
	out_7_bits = ~out_7_bits;
end
endmodule