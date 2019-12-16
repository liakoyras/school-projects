module priority_register(
	input logic clock,
	input logic [2:0] d_i,
	input logic en_i,
	output logic [2:0] q_o
)

logic [2:0] register;

assign q_o = register;

always_ff @(posedge clock) begin

	if(en_i) begin
		register <= d_i;
	end

end

endmodule