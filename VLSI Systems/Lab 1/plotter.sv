module plotter(
	input logic [9:0] rows,
	input logic [9:0] columns,
	output logic [3:0] data_red,
	output logic [3:0] data_green,
	output logic [3:0] data_blue
);

// RGB values generation
always_comb begin
	if(columns >= 219 && columns <= 318) begin
		if(rows >= 140 && rows <= 239) begin
			// Upper left square
			data_red = 4'b1111;
			data_green = 4'b1111;
			data_blue = 4'b1111;
		end else begin
			if(rows >= 240 && rows <= 319) begin
				// Bottom left square
				data_red = 4'b1111;
				data_green = 4'b0000;
				data_blue = 4'b0000;
			end else begin
				// Top-bottom-left margin
				data_red = 4'b0000;
				data_green = 4'b0000;
				data_blue = 4'b0000;
			end
		end
	end else begin
		if(columns >= 319 && columns <= 418) begin
			if(rows >= 140 && rows <= 239) begin
				// Upper right square
				data_red = 4'b0000;
				data_green = 4'b1111;
				data_blue = 4'b0000;
			end else begin
				if(rows >= 240 && rows <= 319) begin
					// Bottom right square
					data_red = 4'b0000;
					data_green = 4'b0000;
					data_blue = 4'b1111;
				end else begin
					// Top-bottom-right margin
					data_red = 4'b0000;
					data_green = 4'b0000;
					data_blue = 4'b0000;
				end
			end
		end else begin
			// Left-Right margin
			data_red = 4'b0000;
			data_green = 4'b0000;
			data_blue = 4'b0000;
		end
	end
end


endmodule