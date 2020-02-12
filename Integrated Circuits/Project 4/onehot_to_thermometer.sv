module onehot_to_thermometer (
	input logic [7:0] A,
	output logic [7:0] Z
);

logic found_1;

always_comb begin
	found_1 = 0;
	
	for(i = 0; i < 8; i = i + 1) begin
		if(A[i])
			found_1 = 1;
		Z[i] = found_1;
	end
end
endmodule