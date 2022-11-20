#!/usr/bin/perl

use strict;
use Math::Round;

my $MATCHES_JSON = " matches-0001-0100.json";
$MATCHES_JSON = @ARGV[0] if @ARGV >= 1;
    
my @match_vals = `jq .[].match_val < $MATCHES_JSON`;


my $numbins = 100;
my @hist;
for (my $i = 0; $i < $numbins; $i++) {
    $hist[$i] = 0;
}

for (@match_vals) {
    chomp;
#    my $i = round(($numbins - 1)*($_ + 1)/2);
    my $i = round(($numbins - 1)*$_);
    next unless 0 <= $i && $i < $numbins;
    $hist[$i]++;
}

for (my $i = 0; $i < $numbins; $i++) {
#    my $v = 2*$i/($numbins - 1) - 1;
    my $v = $i/($numbins - 1);
    my $c = $hist[$i];
    print "$v $c\n";
}
