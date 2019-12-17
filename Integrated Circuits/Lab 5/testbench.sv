module testbench;

// Inputs
logic clock;
logic reset;
logic [7:0] requests;

// Outputs
logic [6:0] low_priority_out;
logic [6:0] requests_count;
logic [7:0] grants_8;

// Generate clock
always begin
	clock = 1;
	#5ns;
	clock = 0;
	#5ns;
end

// Instantiate the buggy arbiter
round_robin r1 (
	.clock(clock),
	.reset(reset),
	.requests(requests),
	.grants_8(grants_8),
	.low_priority_out(low_priority_out),
	.requests_count(requests_count)
);

// Testbench
initial begin
	
	reset <= 0;
	requests <= 4'b0000; // exists=0, grants=0000
	// No bugs
	@(posedge clock);
	
	requests <= 4'b0001; // exists=1, grants=0001

	@(posedge clock);
	reset <= 1;
	requests <= 4'b0010; // exists=1, grants=0010

	@(posedge clock);
	requests <= 4'b0011; // exists=1, grants=0001

	@(posedge clock);
	requests <= 4'b1111; // exists=1, grants=0010
	
	@(posedge clock); // exists=1, grants=0100
	
	@(posedge clock);
	requests <= 4'b1011; // exists=1, grants=1000
	
	// Bug 1 (it just skips when it is 1's turn)
	/* @(posedge clock);
	reset <= 0;
	requests <= 4'b0010; // exists=1, grants=0010
	
	@(posedge clock);
	requests <= 4'b1111; // exists=1, grants=0001 */
	 
	// Bug 2 (if the requests don't change, it doesn't change the grant)
	@(posedge clock);
	reset <= 1;
	requests <= 4'b0001; // exists=1, grants=0001
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b0010; // exists=1, grants=0010
	
	@(posedge clock);
	@(posedge clock);
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b0011; // exists=1, grants=0001
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1111; // exists=1, grants=0010
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1011; // exists=1, grants=1000
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1111; 
	
	@(posedge clock);
	@(posedge clock);
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1110;
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1101;
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1111;
	
	@(posedge clock);
	@(posedge clock);
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1110;
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b0000;
	
	// Bug 4 (when it's 2's turn, it also grants 3)
	/* @(posedge clock);
	reset <= 1;
	requests <= 4'b1111;
	
	@(posedge clock);
	reset <= 0;
	requests <= 4'b1111; // exists=1, grants=0010
	
	@(posedge clock);
	requests <= 4'b0100; // exists=1, grants=0100 */
	
	// Bug 8 (if it gives a grant to 3, it continues to give as long as it asks)
	/* @(posedge clock);
	requests <= 4'b1111; // exists=1, grants=0001
	
	@(posedge clock);
	requests <= 4'b1001; // exists=1, grants=1000
	
	@(posedge clock);
	requests <= 4'b1001; // exists=1, grants=0001
	
	@(posedge clock);
	requests <= 4'b0001; // exists=1, grants=0001 */
end

endmodule