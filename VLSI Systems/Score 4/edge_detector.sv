module edge_detector(
	input logic clk,
	input logic rst,
	input logic signal_in,
	
	output logic falling_edge
	/* output logic rising_edge */
);

logic edge_reg;
initial begin
	edge_reg = 1'b0; 
end

always_ff @(posedge clk, posedge rst) begin
	if(rst) begin 
		edge_reg <= 1'b0; 
	end else begin 
		edge_reg <= signal_in;
	end
end

assign falling_edge = edge_reg & (~signal_in);
/* assign rising_edge = (~edge_reg) & signal_in; */


endmodule