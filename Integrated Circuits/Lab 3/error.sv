module error (
input logic [8:0] X,
input logic [8:0] O,
output logic err
);

always_comb begin
	for (int i = 0; i<9; i++) begin
		if(X[i] && O[i]) begin
			err = 1;
			break;
		end
		else
			err = 0;
	end
end
endmodule