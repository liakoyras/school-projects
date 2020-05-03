module keyboard (
	input clk,
	input rst,
	input kData,
	input kClock,
	output [7:0] character
);

// keyboard clock edge detection
logic clock_rising, clock_falling;
edge_detector edge_clock (.clk(clk), 
						  .rst(rst), 
						  .signal(kClock), 
						  .rising(clock_rising),
						  .falling(clock_falling)
						 );

// keyboard data edge detection			 
logic data_rising, data_falling;
edge_detector edge_data (.clk(clk), 
						  .rst(rst), 
						  .signal(kData), 
						  .rising(data_rising),
						  .falling(data_falling)
						 );


// creating the sampled data sequence
logic [10:0] data_sequence;
logic waveform_state;
always_ff @(posedge clk, negedge rst) begin
	if(~rst) begin
		data_sequence <= 0;
		waveform_state <= 1;
	end else begin
		// calculating current data (based on rising or falling edges)
		if(data_falling) begin
			waveform_state <= 0;
		end else if(data_rising) begin
			waveform_state <= 1;
		end
		// sampling the data
		if(clock_rising) begin
			data_sequence[9:0] <= data_sequence[10: 1];
			data_sequence[10] <= waveform_state;
		end
	end
end

		
// counting when the sequence is over
logic [4:0] period_counter;
always_ff @(posedge clk, negedge rst) begin
	if(~rst) begin
		period_counter <= 0;
	end else begin
		if(clock_falling) begin
			if(period_counter < 10)
				period_counter <= period_counter + 1'b1;
			if(period_counter == 10)
				period_counter <= 0;
		end
	end
end

// send the signal to other modules
always_comb begin
	if(period_counter==10)
		character = data_sequence[8:1];
	else
		character = 8'b00000000;
end

endmodule