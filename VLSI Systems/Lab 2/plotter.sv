module plotter(
	input logic [9:0] rows,
	input logic [9:0] columns,
	input logic [7:0] character,
	output logic [3:0] data_red,
	output logic [3:0] data_green,
	output logic [3:0] data_blue
);

// RGB values generation
always_comb begin
	// pixels outside of the screen are black
	if(columns >= 640 || rows >= 480) begin
		data_red = 4'b0000;
		data_green = 4'b0000;
		data_blue = 4'b0000;
	end else begin
		// plotting logic goes here
		if(character == 8'b00101110) begin // hex: 2d
			data_red = 4'b1111;
			data_green = 4'b0000;
			data_blue = 4'b0000;
		end else begin
			if(character == 8'b00110010) begin // hex: 32
				data_red = 4'b0000;
				data_green = 4'b0000;
				data_blue = 4'b1111;
			end else begin
				if(character == 8'b00110100) begin // hex: 34
					data_red = 4'b0000;
					data_green = 4'b1111;
					data_blue = 4'b0000;
				end else begin
					// if no character is pressed
					data_red = 4'b1111;
					data_green = 4'b1111;
					data_blue = 4'b1111;
				end
			end			
		end
	end
end


endmodule