module m_3_to_8_decoder(
	input logic [2:0] in_3_bits,
	input logic enable,
	output logic [7:0] out_8_bits
);

assign out_8_bits = (!enable)? 8'b00000000:
					(in_3_bits == 3'b000)? 8'b00000001 :
					(in_3_bits == 3'b001)? 8'b00000010 :
					(in_3_bits == 3'b010)? 8'b00000100 : 
					(in_3_bits == 3'b011)? 8'b00001000 : 
					(in_3_bits == 3'b100)? 8'b00010000 :
					(in_3_bits == 3'b101)? 8'b00100000 :
					(in_3_bits == 3'b110)? 8'b01000000 :
										   8'b10000000 ;
										
endmodule