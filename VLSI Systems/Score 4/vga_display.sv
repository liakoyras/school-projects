module vga_display (
	input logic clk,
	input logic rst,
	
	input logic [6:0][5:0][1:0] panel,
	input logic [6:0] play,
	input logic turn,
	
	input logic win,
	input logic [2:0] winner_column,
	input logic [2:0] winner_row,
	input logic [1:0] winner_kind,
	input logic full,
	
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
			 .panel(panel),
			 .play(play),
			 .turn(turn),
			 .win(win),
			 .winner_column(winner_column),
			 .winner_row(winner_row),
			 .winner_kind(winner_kind),
			 .full(full),
			 .data_red(red),
			 .data_green(green),
			 .data_blue(blue)
			);


endmodule