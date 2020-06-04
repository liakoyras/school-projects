module score4 (
	input  logic clk,
	input  logic rst,

	input  logic left,
	input  logic right,
	input  logic put,
	
	output logic player,
	output logic invalid_move,
	output logic win_a,
	output logic win_b,
	output logic full_panel,

	output logic hsync,
	output logic vsync,
	output logic [3:0] red,
	output logic [3:0] green,
	output logic [3:0] blue	
);

// set unused i-o
assign player = 0;
assign invalid_move = 0;
assign win_a = 0;
assign win_b = 0;
assign full_panel = 0;


// transfer state
logic [6:0][5:0][1:0] game_panel;
logic [6:0] game_play;
logic game_turn;

state game_state (.clk(clk),
				.rst(rst),
				.panel(game_panel),
				.play(game_play),
				.turn(game_turn)
				);


vga_display vga (.clk(clk),
			 .rst(rst),
			 .panel(game_panel),
			 .play(game_play),
			 .turn(game_turn),
			 .hsync(hsync),
			 .vsync(vsync),
			 .red(red),
			 .green(green),
			 .blue(blue)
			);



endmodule
