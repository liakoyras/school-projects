module synchronization(
	input logic clk,
	input logic rst,
	output logic hsync,
	output logic vsync,
	output logic [9:0] rows,
	output logic [9:0] columns
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
// columns (columns) and rows counters


always_ff @(posedge clk, negedge rst) begin
	if(~rst) begin
		columns <= 0;
		rows <= 0;
	end else begin
		if(half_clock) begin
			if(columns < 799)
				columns <= columns + 1'b1;
			else begin
				columns <= 0;
				if(rows < 523)
					rows <= rows+1'b1;
				else
					rows <= 0;
			end
		end
	end
end

// HSYNC pulse generation
always_comb begin
	if(columns >= 656 && columns <= 751)
		hsync = 0;
	else
		hsync = 1;
end

// VSYNC pulse generation
/* assign vsync = (rows == 491 && rows == 492) ? 1'b0 : 1'b1; */
always_comb begin
	if(rows >= 491 && rows <= 492)
		vsync = 0;
	else
		vsync = 1;
end


endmodule