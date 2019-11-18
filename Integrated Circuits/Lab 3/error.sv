module error (
input logic [8:0] X,
input logic [8:0] O,
output logic err
);

always_comb
begin
	for (i = 0; i<9; i=i+1)
		if(X[i] && O[i])
			err = 1;
		else
			err = 0;
	end
end
endmodule