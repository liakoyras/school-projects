module testbench;
parameter WIDTH = 4;

logic clk, rst;
logic cnt_enable;
logic [WIDTH-1:0] counter_out;

always begin //Generate the clock, T=10ns
	clk = 1;
	#5ns;
	clk = 0;
	#5ns;
end

counter				
	#(.WIDTH (WIDTH)) //Instantiate a counter
cntr1
	(.clk (clk),
	.reset (rst),
	.satEn (cnt_enable),
	.val_out (counter_out));
	
initial begin
	cnt_enable <= 0; //Set starting values
	rst <= 0;
	
	@(posedge clk); //Reset the counter
	rst <= 1;
	
	@(posedge clk); //Wait
	rst <= 0;
	
	@(posedge clk); //Start counting
	cnt_enable <= 1;
	
	repeat(18) begin //Count until after overflow
		@(posedge clk);
	end
	
	cnt_enable <= 0; //Stop the counting, retain value
	repeat(3) @(posedge clk);
	
	@(negedge clk); //Asynchronous reset with priority over satEn
	rst <= 1;
	cnt_enable <= 1;
	
	@(posedge clk); //Start counting again
	rst <= 0;
	
	repeat(2) @(posedge clk);
end
endmodule