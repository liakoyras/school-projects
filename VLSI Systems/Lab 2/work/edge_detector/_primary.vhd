library verilog;
use verilog.vl_types.all;
entity edge_detector is
    port(
        clk             : in     vl_logic;
        rst             : in     vl_logic;
        \signal\        : in     vl_logic;
        rising          : out    vl_logic;
        falling         : out    vl_logic
    );
end edge_detector;
