module keyboard_tb();

logic clk,kclk,kdata, rst, waveform_state;
logic[0:7] translated_key;
logic [3:0] counter;

always begin
	clk=1;
	#5ns;
	clk=0;
	#5ns;
	end


keyboard test(.clk(clk), 
		    .rst(rst), 
		    .kData(kdata), 
		    .kClock(kclk),
		    .character(translated_key),
			.counter(counter),
			.waveform_state(waveform_state)
	   	   );

initial begin
	repeat(100) @(posedge clk);
	rst <= 1;
	repeat(10) @(posedge clk);
	rst <=0;
	repeat(1000) @(posedge clk);
	rst <= 1;
	kclk<=1;
	kdata<=1;
	$display("Key code given: 11011000");
	repeat(10000) @(posedge clk);
	//1st keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//2nd keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//3rd keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//4th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//5th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//6th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//7th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//8th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//9th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//10th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//11th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1; 
	
	//next key pressed
	
	
	$display("Key code given: 00010011");
	repeat(10000) @(posedge clk);
	//1st keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//2nd keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//3rd keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//4th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//5th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//6th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//7th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//8th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//9th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//10th keyboard clk
	kdata<=0;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1;
	//11th keyboard clk
	kdata<=1;
	repeat(4000) @(posedge clk);
	kclk<=0;
	repeat(4000) @(posedge clk);
	kclk<=1; 
	end

endmodule