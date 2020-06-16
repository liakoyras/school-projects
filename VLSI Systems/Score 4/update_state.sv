module update_state (
	input  logic clk,
	input  logic rst,
	
	input  logic [6:0][5:0][1:0] panel_in,
	input  logic [6:0] play_in,
	input  logic turn_in,
		
	input  logic left,
	input  logic right,
	input  logic put,
	
	input logic win,
	input logic full,
	
	output logic [6:0][5:0][1:0] panel_out,
	output logic [6:0] play_out,
	output logic turn_out,
	
	output logic invalid
);

// passing the inputs through an edge detector in order for it to be independent of the duration of users input
logic l;
logic r;
logic p;

edge_detector ed1 (.clk(clk),
				.rst(rst),
				.signal_in(left),
				.falling_edge(l)
				);

edge_detector ed2 (.clk(clk),
				.rst(rst),
				.signal_in(right),
				.falling_edge(r)
				);
				
edge_detector ed3 (.clk(clk),
				.rst(rst),
				.signal_in(put),
				.falling_edge(p)
				);
				
// checking for free rows
logic [2:0] available_row;
logic valid_column;
free_row row (.panel(panel_in),
			.play(play_in),
			.free(available_row),
			.valid(valid_column)
			);			


// find the selected column
logic [2:0] selected_column;
active_column column (.play(play_in),
					.column(selected_column)
					);

// FSM
enum logic [3:0] {
	WAIT = 4'b0000,
	CIL = 4'b0001,
	ML = 4'b0010,
	CIR = 4'b0011,
	MR = 4'b0100,
	CIP = 4'b0101,
	PUT = 4'b0110,
	INVALID = 4'b0111,
	WIN = 4'b1000,
	FULL = 4'b1111
} fsm_state;

// start the game state from idle
initial begin
	fsm_state = WAIT;
	invalid = 0;
	play_out = 7'b0000001;
	panel_out = 0;
	turn_out = 0;
end

always_ff @(posedge clk) begin
	if(rst) begin
		fsm_state <= WAIT;
		invalid <= 0;
		play_out <= 7'b0000001;
		panel_out <= 0;
		turn_out <= 0;
	end else begin
		case(fsm_state)
			WAIT: begin // default state, waiting for input
				if(l) fsm_state <= CIL;
				else if(r) fsm_state <= CIR;
				else if(p) fsm_state <= CIP;
				else if(win) fsm_state <= WIN;
				else if(full) fsm_state <= FULL;
				else fsm_state <= WAIT;
			end
			CIL: begin // checks if the left move propsed is possible
				if(play_in[0]==1) begin
					fsm_state <= INVALID;
				end 
				else fsm_state <= ML; 
			end
			CIR: begin // checks if the right move proposed is possible
				if(play_in[6]==1) begin
					fsm_state <= INVALID;
				end
				else fsm_state <= MR;
			end
			CIP: begin // checks if the put move proposed is possible (a free row exists in the selected column)
				if(~valid_column) begin
					fsm_state <= INVALID;
				end
				else fsm_state <= PUT;
			end
			ML: begin // moves the selected column to the left
				invalid <= 0;
				play_out <= play_in >> 1;
				fsm_state <= WAIT;
			end
			MR: begin // moves the selected column to the right
				invalid <= 0;
				play_out <= play_in << 1;
				fsm_state <= WAIT;
			end
			PUT: begin // puts a block on the first available row of the selected column and changes the turn to the other player
				invalid <= 0;
				panel_out <= panel_in;
				if(turn_in == 0) begin
					panel_out[selected_column][available_row] <= 2'b01;
				end else begin
					panel_out[selected_column][available_row] <= 2'b10;
				end
				turn_out <= ~turn_in;
				fsm_state <= WAIT;
			end
			INVALID: begin
				invalid <= 1;
				fsm_state <= WAIT;
			end
			WIN: begin
				invalid <= 0;
				fsm_state <= WIN;
			end
			FULL: begin
				fsm_state <= FULL;
			end
		endcase
	end
end

endmodule