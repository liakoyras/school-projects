module testbench;
/* 
 int number_size = 32;
const int index_size = 5;
const int memory_size = 32; */

// Inputs
logic clock;
logic reset;
logic start_in;
logic [31:0] target_in;

// Outputs
logic [4:0] index_out;

binary_search #(32, 5, 32) search
(
	.clk(clock),
	.rst(reset),
	.start(start_in),
	.target(target_in),
	.out(index_out)
);

// Generate clock
always begin
	clock = 1;
	#5ns;
	clock = 0;
	#5ns;
end


initial begin
	reset <=1;
	
	@(posedge clock);
	reset <= 0;
	start_in <= 1;
	target_in <= 15;
	
	@(posedge clock);
	start_in <= 0;
end

endmodule