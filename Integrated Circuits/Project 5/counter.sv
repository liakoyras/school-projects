module counter
#(parameter int WIDTH = 4)
//input -Output List
(input logic clk,
input  logic  reset,
input logic satEn,
output logic[WIDTH-1:0] val_out);
//Internal Signal(flip-flops)
logic [WIDTH-1:0] count;
//Functionality
always_ff @(posedge clk, posedge reset) begin
    if(reset) count <= 0;
    else begin
        if(satEn) count <= count + 1;
    end
end
assign val_out = count;
endmodule
