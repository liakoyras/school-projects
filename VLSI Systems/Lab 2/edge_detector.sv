module edge_detector (
	input logic clk,
	input logic rst,
	input logic signal,
	output logic rising,
	output logic falling
);

// synchronization
logic x, y, z;
always_ff @(posedge clk, negedge rst) begin
	if(~rst) begin
		x <= 0;
		y <= 0;
		z <= 0;
	end else begin
		x <= signal;
		y <= x;
		z <= y;
	end
ends

// edge detection
assign rising = y & (~z);
assign falling = z & (~y);

endmodule