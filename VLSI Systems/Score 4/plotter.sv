module plotter(
	input logic [9:0] rows,
	input logic [9:0] columns,
	
	input logic [6:0][5:0][1:0] panel,
	input logic [6:0] play,
	input logic turn,
	
	input logic win,
	input logic [2:0] winner_column,
	input logic [2:0] winner_row,
	input logic [1:0] winner_kind,
	input logic full,

	output logic [3:0] data_red,
	output logic [3:0] data_green,
	output logic [3:0] data_blue
);

logic [7:0] a;
logic [6:0] b;
logic [6:0] c;
logic [5:0] d;
logic [5:0] z;
logic [4:0] k;

assign a = 90;
assign b = 40;
assign c = 30;
assign d = 20;
assign z = 20;
assign k = 10;

// RGB values generation
always_comb begin
	// pixels outside of the screen are black
	if(columns >= 640 || rows >= 480) begin
		data_red = 4'b0000;
		data_green = 4'b0000;
		data_blue = 4'b0000;
	end else begin
		// l-r & top-bot margin (the first 2 conditions are the l-r and the rest the top, bottom and between the selected line and the grid)
		if(columns <= a || columns >= (640 - a) || rows <= d || rows >= (480-z) || (rows <= (480 - z - k) && rows >= (d + 6*b + 5*c))) begin
			data_red = 4'b0000;
			data_green = 4'b0000;
			data_blue = 4'b0000;
		end else begin
			if(rows >= (2*d + 6*b + 5*c + z) && rows <= (480 - z)) begin // defines the selected column  line
				if(~win && ~full) begin
					data_red = 4'b0000; // sets the default color
					data_green = 4'b0000;
					data_blue = 4'b0000;
					for(int i=0; i<7; i=i+1) begin // runs through each "selected column" box
						if((columns <= (a + ((i+1)*b) + (i*c))) && (columns >= (a + (i*b) + (i*c)))) begin // bounds each box
							if(play[i] == 1'b1) begin // colors the active column
								if(turn == 0) begin // selects its color depending on the player
									data_red = 4'b1111;
									data_green = 4'b0000;
									data_blue = 4'b0000;
								end else begin
									data_red = 4'b0000;
									data_green = 4'b1111;
									data_blue = 4'b0000;
								end
							end else begin // what happens if the box is not selected
								data_red = 4'b0000;
								data_green = 4'b0000;
								data_blue = 4'b0000;
							end
						end
					end
				end else if(full) begin // if the board is full, color the bar blue
					data_red = 4'b0000;
					data_green = 4'b0000;
					data_blue = 4'b1111;
				end else begin // if someone won set the color of the line to the winner's color
					if(turn) begin // red won
						data_red = 4'b1111;
						data_green = 4'b0000;
						data_blue = 4'b0000;
					end else begin // green won
						data_red = 4'b0000;
						data_green = 4'b1111;
						data_blue = 4'b0000;
					end
				end
			end else begin // the game panel
				data_red = 4'b0000; // sets the default color
				data_green = 4'b0000;
				data_blue = 4'b0000;
				for(int i=0; i<7; i=i+1) begin // runs through each column
					for(int j=0; j<6; j = j+1)begin // runs through each row
						if((columns <= (a + ((i+1)*b) + (i*c))) && (columns >= (a + (i*b) + (i*c))) && (rows <= (d + ((j+1)*b) + (j*c))) && (rows >= (d + (j*b) + (j*c)))) begin // bounds each box	
							if(~win) begin // default colors, if no win
								if(panel[i][j] == 2'b01) begin // selects what color is each square
									data_red = 4'b1111;
									data_green = 4'b0000;
									data_blue = 4'b0000;
								end else if (panel[i][j] == 2'b10) begin
									data_red = 4'b0000;
									data_green = 4'b1111;
									data_blue = 4'b0000;
								end else begin
									data_red = 4'b0000;
									data_green = 4'b0000;
									data_blue = 4'b0000;
								end							
							end else begin
								if(winner_kind == 2'b00) begin // horizontal win
									if(j == winner_row && ((i == winner_column)||(i == winner_column+1)||(i == winner_column+2)||(i == winner_column+3))) begin// winning colors
										if(turn) begin // red won
											data_red = 4'b1010;
											data_green = 4'b0000;
											data_blue = 4'b1010;
										end else begin // green won
											data_red = 4'b0000;
											data_green = 4'b1010;
											data_blue = 4'b1010;
										end
									end else begin
										if(panel[i][j] == 2'b01) begin // non-winning colors
											data_red = 4'b1111;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end else if (panel[i][j] == 2'b10) begin
											data_red = 4'b0000;
											data_green = 4'b1111;
											data_blue = 4'b0000;
										end else begin
											data_red = 4'b0000;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end		
									end
								end else if(winner_kind == 2'b01) begin // vertical win
									if(i == winner_column && ((j == winner_row)||(j == winner_row+1)||(j == winner_row+2)||(j == winner_row+3))) begin// winning colors
										if(turn) begin // red won
											data_red = 4'b1010;
											data_green = 4'b0000;
											data_blue = 4'b1010;
										end else begin // green won
											data_red = 4'b0000;
											data_green = 4'b1010;
											data_blue = 4'b1010;
										end
									end else begin
										if(panel[i][j] == 2'b01) begin // non-winning colors
											data_red = 4'b1111;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end else if (panel[i][j] == 2'b10) begin
											data_red = 4'b0000;
											data_green = 4'b1111;
											data_blue = 4'b0000;
										end else begin
											data_red = 4'b0000;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end		
									end
								end else if(winner_kind == 2'b10) begin // diagonal 1 win
									if((j == winner_row && i == winner_column)||(j == winner_row+1 && i == winner_column+1)||(j == winner_row+2 && i == winner_column+2)||(j == winner_row+3 && i == winner_column+3)) begin// winning colors
										if(turn) begin // red won
											data_red = 4'b1010;
											data_green = 4'b0000;
											data_blue = 4'b1010;
										end else begin // green won
											data_red = 4'b0000;
											data_green = 4'b1010;
											data_blue = 4'b1010;
										end
									end else begin
										if(panel[i][j] == 2'b01) begin // non-winning colors
											data_red = 4'b1111;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end else if (panel[i][j] == 2'b10) begin
											data_red = 4'b0000;
											data_green = 4'b1111;
											data_blue = 4'b0000;
										end else begin
											data_red = 4'b0000;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end		
									end
								end else begin // diagonal 2 win
									if((j == winner_row && i == winner_column)||(j == winner_row-1 && i == winner_column+1)||(j == winner_row-2 && i == winner_column+2)||(j == winner_row-3 && i == winner_column+3)) begin// winning colors
										if(turn) begin // red won
											data_red = 4'b1010;
											data_green = 4'b0000;
											data_blue = 4'b1010;
										end else begin // green won
											data_red = 4'b0000;
											data_green = 4'b1010;
											data_blue = 4'b1010;
										end
									end else begin
										if(panel[i][j] == 2'b01) begin // non-winning colors
											data_red = 4'b1111;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end else if (panel[i][j] == 2'b10) begin
											data_red = 4'b0000;
											data_green = 4'b1111;
											data_blue = 4'b0000;
										end else begin
											data_red = 4'b0000;
											data_green = 4'b0000;
											data_blue = 4'b0000;
										end		
									end
								end
							end
						end
					end
				end
			end
		end
	end
end


endmodule