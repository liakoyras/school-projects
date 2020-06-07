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
assign invalid_move = 0;


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


// display the game
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
			
// output the current player
assign player = game_turn;
			
// check if the board is full_panel
full_board full (.panel(game_panel),
				 .full(full_panel)
				);


logic game_winner;
logic winner_exists;
// find if there is a winner
find_winner winner (.panel(game_panel),
					.turn(game_turn),
					.winner(game_winner),
					.exists(winner_exists)
				   );
					
assign win_a = winner_exists & (~game_winner);
assign win_b = winner_exists & game_winner;


endmodule
