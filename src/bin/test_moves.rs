use oxi::moves::get_all_possible_moves;

fn main() {
    let moves = get_all_possible_moves();
    println!("Total moves generated: {}", moves.len());
    
    // Count queen and knight moves separately
    let mut queen_knight = 0;
    let mut promotions = 0;
    
    for mv in &moves {
        if mv.len() == 5 {
            promotions += 1;
        } else {
            queen_knight += 1;
        }
    }
    
    println!("Queen and knight moves: {}", queen_knight);
    println!("Pawn promotions: {}", promotions);
    
    // Check for duplicates
    let mut unique = std::collections::HashSet::new();
    for mv in &moves {
        if !unique.insert(mv) {
            println!("Duplicate move found: {}", mv);
        }
    }
    println!("Unique moves: {}", unique.len());
    
    // Print some samples
    println!("\nFirst 10 moves: {:?}", &moves[..10]);
    println!("\nLast 10 moves: {:?}", &moves[moves.len()-10..]);
}