 module PanelDisplay(
input logic clk,
input logic rst,
output logic hsync,
output logic vsync,
output logic [3:0] red,
output logic [3:0] green,
output logic [3:0] blue
);

// Divide clock by 2
logic half_clock;
always_ff @(posedge clk, negedge rst) begin
	if(~rst)
		half_clock <= 1'b0;
	else
		half_clock <= ~half_clock;
end

// Counters
// Pixels (columns) and rows counters
logic [9:0] pixels;
logic [9:0] rows;

always_ff @(posedge half_clock, negedge rst) begin
	if(~rst) begin
		pixels <= 0;
		rows <= 0;
	end else begin
		if(pixels < 799)
			pixels <= pixels + 1'b1;
		else begin
			pixels <= 0;
			if(rows < 523)
				rows <= rows+1'b1;
			else
				rows <= 0;
		end
	end
end

// HSYNC pulse generation
always_ff @(posedge half_clock, negedge rst) begin
	if(~rst) 
		hsync <= 1;
	else begin
		if(pixels >= 656 && pixels <= 751)
			hsync <= 0;
		else
			hsync <= 1;
	end
end

// VSYNC pulse generation
always_ff @(posedge half_clock, negedge rst) begin
	if(~rst) 
		vsync <= 1;
	else begin
		if(pixels >= 491 && pixels <= 492)
			vsync <= 0;
		else
			vsync <= 1;
	end
end

// RGB generation
always_ff @(posedge half_clock, negedge rst) begin
	if(~rst) begin
		red <= 4'b0000;
		green <= 4'b0000;
		blue <= 4'b0000;
	end else begin
		if(pixels >= 219 && pixels <= 318) begin
			if(rows >= 140 && rows <= 239) begin
				// Upper left square
				red <= 4'b1111;
				green <= 4'b1111;
				blue <= 4'b1111;
			end else begin
				if(rows >= 240 && rows <= 319) begin
					// Bottom left square
					red <= 4'b1111;
					green <= 4'b0000;
					blue <= 4'b0000;
				end else begin
					// Top-bottom-left margin
					red <= 4'b0000;
					green <= 4'b0000;
					blue <= 4'b0000;
				end
			end
		end else begin
			if(pixels >= 319 && pixels <= 418) begin
				if(rows >= 140 && rows <= 239) begin
					// Upper right square
					red <= 4'b0000;
					green <= 4'b1111;
					blue <= 4'b0000;
				end else begin
					if(rows >= 240 && rows <= 319) begin
						// Bottom right square
						red <= 4'b0000;
						green <= 4'b0000;
						blue <= 4'b1111;
					end else begin
						// Top-bottom-right margin
						red <= 4'b0000;
						green <= 4'b0000;
						blue <= 4'b0000;
					end
				end
			end else begin
				// Left-Right margin
				red <= 4'b0000;
				green <= 4'b0000;
				blue <= 4'b0000;
			end
		end
	end
end


endmodule