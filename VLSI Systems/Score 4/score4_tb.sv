module score4_tb;

// set VGA_LOG = 0 to disable logging of the VGA output
parameter logic VGA_LOG = 1;

// TESTBENCH VARIABLES
integer fileout;
integer frame_cnt;
logic write_frame;
string s;

enum {WIN_1_D, WIN_1_R, WIN_1_C, WIN_2_D, WIN_2_R, WIN_2_C,
	  H_ERROR, V_ERROR, FULL, MY_GAME
	  } scenario;


// DUT SIGNALS
logic clk;
logic rst;

logic left;
logic right;
logic put;

logic player;
logic invalid_move;
logic win_a;
logic win_b;
logic full_panel;

logic vga_hsync;
logic vga_vsync;
logic [3:0] vga_red;
logic [3:0] vga_green;
logic [3:0] vga_blue;


// DUT INSTANTIATION
score4 DUT(
	.clk   		  (clk),
	.rst   		  (rst),
	.left  		  (left),
	.right 		  (right),
	.put   		  (put),
	.player       (player),
	.invalid_move (invalid_move),
	.win_a 		  (win_a),
	.win_b 		  (win_b),
	.full_panel   (full_panel),
	.hsync 		  (vga_hsync),
	.vsync 		  (vga_vsync),
	.red   		  (vga_red),
	.green 		  (vga_green),
	.blue  		  (vga_blue)
);


// 50MHz CLOCK GENERATION
always begin
	clk = 0;
	#10ns;
	clk = 1;
	#10ns;
end


// STIMULUS
initial begin
	$timeformat(-9, 0, " ns", 6);

	// POSITIVE RESET
	RESET();
	$display("\n\n~~~~~~~~ GAME STARTS ~~~~~~~~");
		
	//select one of the following scenarios:
	//	WIN_1_D: player 1 wins (diagonal)
	//	WIN_1_R: player 1 wins (row)
	//	WIN_1_C: player 1 wins (column)
	//	WIN_2_D: player 2 wins (diag)
	//  WIN_2_R: player 2 wins (row)
	//	WIN_2_C: player 2 wins (column)
	//  H_ERROR: check for errors on rows
	//  V_ERROR: check for errors on columns
	//  FULL:    Fill the panel -- WARNING! this creates 70 frames ~> almost 2GB
	//  MY_GAME: create your own game 
	scenario = WIN_1_D;
	
	case(scenario)
	MY_GAME: begin 
		lets_put();
		lets_put();
		go_right();
		go_right();
		lets_put();
		// YOUR GAME HERE
		// Choose your moves: go_right(), go_left(), lets_put()
		
		// example code
		/* lets_put();
		go_right();
		go_right();
		lets_put();
		go_left();
		lets_put(); */
		// end of example
	
	
		end // end MY_GAME
		
	WIN_1_D:
		p1_wins_d();
		
	WIN_1_R:
		p1_wins_r();
		
	WIN_1_C:
		p1_wins_c();
	
	WIN_2_D: 
		p2_wins_d();
		
	WIN_2_R: 
		p2_wins_r();
	
	WIN_2_C: 
		p2_wins_c();
		
	H_ERROR: 
		check_h_error();
		
	V_ERROR: 
		check_v_error();
		
	FULL: 
		check_full();
	
	default 
		$display("You didn't choose a valid scenario!");
	endcase
	
	
	$display("~~~~~~~~~ GAME OVER ~~~~~~~~~");
		
	$finish;
end

// End of the main body of the testbench
// ATTENTION: In the next lines you can find the implementation 
//            for the tasks used in the testbench.


// RESET TASK
task RESET();
	scenario <= MY_GAME;
	write_frame <= 0;
	frame_cnt <= 0;
	
	left <= 0;
	right <= 0;
	put <= 0;
	
	rst <= 1;
	repeat(2) @(posedge clk);
	rst <= 0;
	repeat(10) @(posedge clk);
endtask


// Choose action: go left
task go_left();
		
	$display("Moving Left...");
	make_move(1'b0, 1'b0, 1'b1);
endtask


// Choose action: go right
task go_right();
	
	$display("Moving Right...");
	make_move(1'b0, 1'b1, 1'b0);
endtask


// Choose action: put
task lets_put();
	
	$display("Placing Token!");
	make_move(1'b1, 1'b0, 1'b0);
endtask

// Choose action: do_nothing
task do_nothing();
	
	$display("Nothing for this frame.");
	make_move(1'b0, 1'b0, 1'b0);
endtask


// Select the next action
task make_move(input logic p, r, l);	
	
	put <= p;
	right <= r;
	left <= l;
	
	repeat(10) @(posedge clk);
	
	put <= 0;
	right <= 0;
	left <= 0;
	
	
	write_frame <= 1;
	
	@(negedge write_frame);
	@(posedge clk);
		
endtask


// Case study where player 1 wins (diagonal)
task p1_wins_d;
	lets_put();
	go_right();
	
	repeat(2) begin
		lets_put();
		lets_put();
		go_right();
	end
	
	lets_put();
	go_left();
	repeat(2) lets_put();
	go_right();
	repeat(3) lets_put();
endtask


// Case study where player 1 wins (row)
task p1_wins_r;
	repeat(3) begin
		repeat(2) lets_put();
		go_right();
	end
	
	lets_put();
	
endtask


// Case study where player 1 wins (column)
task p1_wins_c;
	repeat(3) begin
		lets_put();
		go_right();
		lets_put();
		go_left();
	end
	lets_put();
	
endtask


// Case study where player 2 wins (diagonal)
task p2_wins_d;
	lets_put();
		
	repeat(2) begin
		go_right();
		lets_put();
	end
		
	lets_put();
		
	go_right();
	repeat(2) lets_put();
		
	go_right();
	lets_put();
		
	go_left();
	lets_put();
		
	go_right();
	repeat(2) lets_put();
		
	go_left();
	lets_put();
	
	go_right();
	lets_put();
endtask


// Case study where player 1 wins (row)
task p2_wins_r;
	repeat(3) begin
		repeat(2) lets_put();
		go_right();
	end
	repeat(2) begin
		go_right();
		lets_put();
		go_left();
		lets_put();
	end

endtask


// Case study where player 2 wins (column)
task p2_wins_c;
	repeat(2) begin
		lets_put();
		go_right();
		lets_put();
		go_left();
	end
	repeat(2) begin
		lets_put();
		go_right();
	end

	lets_put();
	go_left();
	lets_put();
endtask


// Case study for locating errors in a row
task check_h_error;
	lets_put();
	go_left();
	lets_put();
	repeat(8) begin
		go_right();
		lets_put();
	end
		
	go_left();
	lets_put();
		
	go_right();
	lets_put();
endtask


// Case study for locating erros in a column
task check_v_error;
	repeat(7) lets_put();		
	go_right();
	lets_put();
endtask


// Case study for checking a full game panel
task check_full;
	lets_put();
	repeat(6) begin
		go_right();
		lets_put();
	end
	lets_put();
	repeat(6) begin
		go_left();
		lets_put();
	end
	lets_put();
	repeat(6) begin
		go_right();
		lets_put();
	end
	
	repeat(2) begin
		go_left();
		lets_put();
		go_right();
		lets_put();
		lets_put();
		go_left();
		lets_put();
		lets_put();
		go_right();
		lets_put();
	
		repeat(2) go_left();
	end
	
	go_left();
	repeat(3) lets_put();
	go_left();
	repeat(2) lets_put();
	repeat(2) go_right();
	repeat(3) lets_put();
	repeat(2) go_left();
	lets_put();
	
endtask


// MONITORS THE OUTPUTS OF THE DESIGN
initial begin
	$monitor("player : %b , invalid_move : %b , win_a : %b , win_b : %b , full_panel : %b\n", player, invalid_move, win_a, win_b, full_panel);
end


// Write frame to VGA log
always @(negedge vga_vsync) begin
	if ( write_frame ) begin
		if (VGA_LOG==1) begin
			s.itoa(frame_cnt);	
			fileout = $fopen({"vga_frame_", s, ".txt"});
		
			repeat (838400) begin
				@(posedge clk);
				$fdisplay(fileout, "%t: %b %b %b %b %b", $time, vga_hsync, vga_vsync, vga_red, vga_green, vga_blue);
			end
			@(negedge clk); 
	
			frame_cnt ++;
			$fclose(fileout);
		end
		else begin
			repeat (838400) @(posedge clk);
			@(negedge clk);
		end
	write_frame <= 0;
	end
	

end



endmodule
