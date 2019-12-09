module binary_search 
#(
	parameter int number_size,	// The size of the numbers stored inside the memory
	parameter int index_size,	// The size of the index of the last element
	parameter int memory_size	// The size of the memory array
)
(
	// Inputs
	input logic clk,
	input logic rst,
	input logic start,
	input logic [number_size-1:0] target,
	input logic [number_size-1:0] memory_in,
	
	// Outputs
	output logic [index_size-1:0] out,
	output logic [index_size-1:0] memory_address
);

enum logic [3:0] {
	IDLE = 4'b0000,
	S0 = 4'b0001,
	S1 = 4'b0010,
	S2 = 4'b0011,
	S3 = 4'b0100,
	S4 = 4'b0101,
	S5 = 4'b0110,
	S6 = 4'b0111,
	S7 = 4'b1000,
	lgr = 4'b1001,
	ml2 = 4'b1010,
	tlt = 4'b1011,
	tgt = 4'b1100
} state;

logic [index_size-1:0] index;
logic [number_size-1:0] left, right, mid;
logic [index_size-1:0] count;
logic [number_size-1:0] tmp;

always_ff @(posedge clk) begin
	if(rst) begin
		state <= IDLE;
	end else begin
		case(state)
			IDLE: begin
				if(start) state <= S0;
				else state <= IDLE;
			end
			S0: begin
				left <= 0;
				right <= memory_size-1;
				index <= -1;
				state <= lgr;
			end
			lgr: begin
				if(left > right) state <= S7;
				else state <= S1;
			end
			S1: begin
				mid <= left - right;
				count <= 0;
				state <= ml2;
			end
			ml2: begin
				if(mid < 2) state <= S2;
				else state <= S3;
			end
			S2: begin
				count <= count + 1;
				mid = mid - 2;
				state <= ml2;
			end
			S3: begin
				tmp <= memory_in;
				state <= tlt;
			end
			tlt: begin
				if(tmp < target) state <= S4;
				else state <= tgt;
			end
			S4: begin
				left <= count + 1;
				state <= lgr;
			end
			tgt: begin
				if(tmp > target) state <= S5;
				else state <= S6;
			end
			S5: begin
				right <= count -1;
				state <= lgr;
			end
			S6: begin
				index <= tmp;
				state <= lgr;
			end
			S7: begin
				out <= index;
				state <= IDLE;
			end
		endcase
	end
end

assign memory_address = count;

endmodule