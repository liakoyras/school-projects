module full (
input logic [8:0] X,
input logic [8:0] O,
output logic isFull/* ,
output logic [8:0] temp */
);

logic [8:0] temp;
assign temp = (X | O);


always_comb
begin
	if(temp == 9'b111111111)
		isFull = 1;
	else
		isFull = 0;
end
endmodule
