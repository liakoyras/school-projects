library verilog;
use verilog.vl_types.all;
entity plotter is
    port(
        rows            : in     vl_logic_vector(9 downto 0);
        columns         : in     vl_logic_vector(9 downto 0);
        character       : in     vl_logic_vector(7 downto 0);
        data_red        : out    vl_logic_vector(3 downto 0);
        data_green      : out    vl_logic_vector(3 downto 0);
        data_blue       : out    vl_logic_vector(3 downto 0)
    );
end plotter;
