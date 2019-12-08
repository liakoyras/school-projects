module memory #(
	parameter int bits,
	parameter int size,
	parameter int address_size
)
(
	input logic clk, // Clock
/* 	input logic we, // write enable */
	input logic [address_size-1:0] address, // address vector
/* 	input logic [bits:0] dataIn, */
	output logic [bits-1:0] dataOut
);
logic [size-1:0] [bits-1:0] ram;

always_ff @(posedge clk) begin
/* 	if (we)
		ram[address] <= dataIn;
	else */
		dataOut <= ram[address];
end

initial begin
	static int i = 0;

	while(i < size) begin
		ram[i] = i;
		i = i + 1;
	end
end

endmodule