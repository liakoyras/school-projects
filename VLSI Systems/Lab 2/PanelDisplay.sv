module PanelDisplay (
	input logic clk,
	input logic rst,
	input logic char,
	output logic hsync,
	output logic vsync,
	output logic [3:0] red,
	output logic [3:0] green,
	output logic [3:0] blue
);

logic [9:0] data_cols;
logic [9:0] data_rows;


synchronization sync (.clk(clk), 
					  .rst(rst), 
					  .hsync(hsync), 
					  .vsync(vsync),
					  .rows(data_rows),
					  .columns(data_cols)
					 );

plotter plt (.rows(data_rows),
			 .columns(data_cols),
			 .character(char)
			 .data_red(red),
			 .data_green(green),
			 .data_blue(blue)
			);


endmodule