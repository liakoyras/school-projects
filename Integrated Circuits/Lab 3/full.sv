module full (
input logic [8:0] X,
input logic [8:0] O,
output logic isFull
);

logic temp[8:0];
assign temp = X || O;

assign isFull = &temp;

/* always_comb
begin
	if(temp == 9’b111111111)
		isFull = 1;
	else
		isFull = 0;
end */
endmodule
