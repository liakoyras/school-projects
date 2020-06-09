module find_winner(
	input logic [6:0][5:0][1:0] panel,
	input logic turn,

	output logic winner,
	output logic exists
);

// when it is the turn of a player, we check if the other player made a winning move on the previous turn
logic [1:0] player;
always_comb begin
	if(turn==1'b0) begin
		player = 2'b10;
	end else begin
		player = 2'b01;
	end
end

logic [5:0][3:0] exists_horizontal;
logic win_h;

logic [6:0][2:0] exists_vertical;
logic win_v;

logic [3:0][2:0] exists_diagonal_1;
logic win_d1;

logic [3:0][2:0] exists_diagonal_2;
logic win_d2;

always_comb begin
	winner = ~turn; // the winner, if exists, is the previous player
	
	// checking each row for a win
	for(int i=0; i<6; i=i+1) begin 
		for(int j = 0; j<4; j=j+1) begin
			if(panel[j][i] == player && panel[j+1][i] == player && panel[j+2][i] == player && panel[j+3][i] == player) begin
				exists_horizontal[i][j] = 1; // if winner exists on some horizontal combination
			end else begin
				exists_horizontal[i][j] = 0;
			end
		end
	end
	
	win_h = |exists_horizontal; // a horizontal win is confirmed because at least one combination is winning
	
	// checking each column for a win
	for(int k=0; k<7; k=k+1) begin 
		for(int l = 0; l<3; l=l+1) begin
			if(panel[k][l] == player && panel[k][l+1] == player && panel[k][l+2] == player && panel[k][l+3] == player) begin
				exists_vertical[k][l] = 1; // if winner exists on some vertical combination
			end else begin
				exists_vertical[k][l] = 0;
			end
		end
	end
	
	win_v = |exists_vertical; // a vertical win is confirmed
	
	// checking downwards (\) diagonals	for a win
	for(int m=0; m<4; m=m+1) begin
		for(int n = 0; n<3; n=n+1) begin
			if(panel[m][n] == player && panel[m+1][n+1] == player && panel[m+2][n+2] == player && panel[m+3][n+3] == player) begin
				exists_diagonal_1[m][n] = 1; // if winner exists on some diagonal (\) combination
			end else begin
				exists_diagonal_1[m][n] = 0;
			end
		end
	end
	
	win_d1 = |exists_diagonal_1; //  diagonal (\) win is confirmed
	
	// checking upwards (/) diagonals for a win
	for(int p=0; p<4; p=p+1) begin
		for(int r = 3; r<6; r=r+1) begin
			if(panel[p][r] == player && panel[p+1][r-1] == player && panel[p+2][r-2] == player && panel[p+3][r-3] == player) begin
				exists_diagonal_2[p][r-3] = 1; // if winner exists on some diagonal (/) combination
			end else begin
				exists_diagonal_2[p][r-3] = 0;
			end
		end
	end
	
	win_d2 = |exists_diagonal_2; //  diagonal (/) win is confirmed
	
	exists = win_h | win_v | win_d1 | win_d2; // if a winning combination appears on rows, columns or diagonals, a winner is confirmed
end


endmodule