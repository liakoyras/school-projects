module m_7_segment_decoder_4 (
	input logic [3:0] in_4_bits,
	output logic [6:0] out_7_bits
);

always_comb begin
	case (in_4_bits)
		4'b0000: out_7_bits = 7'b0111111;
		4'b0001: out_7_bits = 7'b0000110;
		4'b0010: out_7_bits = 7'b1011011;
		4'b0011: out_7_bits = 7'b1001111;
		4'b0100: out_7_bits = 7'b1100110;
		4'b0101: out_7_bits = 7'b1101101;
		4'b0110: out_7_bits = 7'b1111101;
		4'b0111: out_7_bits = 7'b0000111;
		4'b1000: out_7_bits = 7'b1111111;
		default: out_7_bits = 7'b0000000;
	endcase
	out_7_bits = ~out_7_bits;
end

endmodule