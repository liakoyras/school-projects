module PanelDisplay (
	input logic clk,
	input logic rst,
	input logic character,
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


// remembers the last value someone entered on the keyboard					 
logic [7:0] last_button_pressed;
always_ff @(posedge clk, negedge rst) begin
	if(~rst) begin
		last_button_pressed <= 8'b00000000;
	end else begin
		if(character != 8'b00000000) begin
			last_button_pressed <= character;
		end
	end
end

plotter plt (.rows(data_rows),
			 .columns(data_cols),
			 .character(last_button_pressed),
			 .data_red(red),
			 .data_green(green),
			 .data_blue(blue)
			);


endmodule