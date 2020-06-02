library verilog;
use verilog.vl_types.all;
entity keyboard is
    port(
        clk             : in     vl_logic;
        rst             : in     vl_logic;
        kData           : in     vl_logic;
        kClock          : in     vl_logic;
        character       : out    vl_logic_vector(7 downto 0);
        waveform_state  : out    vl_logic;
        counter         : out    vl_logic_vector(3 downto 0)
    );
end keyboard;
