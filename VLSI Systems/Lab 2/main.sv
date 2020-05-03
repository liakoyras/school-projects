module main (
	input logic clk,
	input logic rst,
	input logic kClock,
	input logic kData,
	output logic hsync,
	output logic vsync,
	output logic [3:0] red,
	output logic [3:0] green,
	output logic [3:0] blue
);

// keyboard driver
logic [7:0] character;
keyboard kb (.clk(clk), 
		    .rst(rst), 
		    .kData(kData), 
		    .kClock(kClock),
		    .character(character)
	   	   );


// vga display driver
PanelDisplay vga (.clk(clk), 
				  .rst(rst), 
				  .character(character), 
				  .hsync(hsync),
				  .vsync(vsync),
				  .red(red),
				  .green(green),
				  .blue(blue)
				 );


endmodule