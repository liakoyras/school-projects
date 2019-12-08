module arbiter_testbench;

// Inputs
logic clock;
logic reset;
logic [3:0] requests;

// Outputs
logic [3:0] grants;
logic grant_exists;

// Generate clock
always begin
	clock = 1;
	#5ns;
	clock = 0;
	#5ns;
end

// Instantiate the buggy arbiter
rr_arbiter_buggy #(4, 8) arbiter (
	.clk(clock),
	.rst(reset),
	.reqs_i(requests),
	.grants_o(grants),
	.any_grant_o(grant_exists)
);

// Testbench
initial begin
	
	reset <= 1;
	requests <= 4'b0000; // exists=0, grants=0000
	// No bugs
	@(posedge clock);
	reset <= 0;
	requests <= 4'b0001; // exists=1, grants=0001

	@(posedge clock);
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
	/* @(posedge clock);
	reset <= 0;
	requests <= 4'b0001; // exists=1, grants=0001
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b0010; // exists=1, grants=0010
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b0011; // exists=1, grants=0001
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1111; // exists=1, grants=0010
	
	@(posedge clock);
	@(posedge clock);
	requests <= 4'b1011; // exists=1, grants=1000 */
	
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
	@(posedge clock);
	requests <= 4'b1111; // exists=1, grants=0001
	
	@(posedge clock);
	requests <= 4'b1001; // exists=1, grants=1000
	
	@(posedge clock);
	requests <= 4'b1001; // exists=1, grants=0001
	
	@(posedge clock);
	requests <= 4'b0001; // exists=1, grants=0001
end

endmodule