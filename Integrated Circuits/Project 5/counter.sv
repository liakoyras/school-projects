module counter
#(parameter int WIDTH = 4) //Seting up variable width

(input logic clk, //Input - Output List
input  logic reset,
input logic satEn,
output logic [WIDTH-1:0] val_out);

logic [WIDTH-1:0] count; //Internal Signal(flip-flops)

//Functionality
always_ff @(posedge clk, posedge reset) begin
    if(reset) count <= 0;
    else begin
        if(satEn) count <= count + 1;
    end
end
assign val_out = count;
endmodule
