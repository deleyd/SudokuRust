use rand::{Rng, SeedableRng};


fn main() {
    let mut xyzzy: [i32; 5] = [1, 2, 3, 4, 5];
    xyzzy[2] = 0;

    let line:&str = "+---+---+---+";
    let _middle:&str = "|...|...|...|";
    // Declares a 3x4 2D array of characters, initialized with 'X'
    let mut grid: [[char; 13]; 13] = [['X'; 13]; 13];

    // Accessing and modifying elements
    grid[0][0] = 'A';
    grid[1][2] = 'B';
    let letters: Vec<char> = line.chars().collect();
    let line2: Vec<char> = _middle.chars().collect();
    grid[0] = letters.clone().try_into().expect("REASON");
    grid[1] = line2.clone().try_into().expect("REASON");
    grid[2] = line2.clone().try_into().expect("REASON");
    grid[3] = line2.clone().try_into().expect("REASON");
    grid[4] = letters.clone().try_into().expect("REASON");
    grid[5] = line2.clone().try_into().expect("REASON");
    grid[6] = line2.clone().try_into().expect("REASON");
    grid[7] = line2.clone().try_into().expect("REASON");
    grid[8] = letters.clone().try_into().expect("REASON");
    grid[9] = line2.clone().try_into().expect("REASON");
    grid[10] = line2.clone().try_into().expect("REASON");
    grid[11] = line2.clone().try_into().expect("REASON");
    grid[12] = letters.clone().try_into().expect("REASON");
    // Iterating and printing
    for row in &grid {
        for &c in row {
            print!("{}", c);
        }
        println!();
    }

    // Construct board to be solved
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(10);
    println!("Random u32: {}", rng.random::<u32>());

    // Top element is current state of the board
    //Stack<int[]> stateStack = new Stack<int[]>();
    let mut state_stack: Vec<[i32; 9*9]> = Vec::new(); // Explicitly typed for clarity

    // Top elements are (row, col) of cell which has been modified compared to previous state
    //Stack<int> rowIndexStack = new Stack<int>();
    //Stack<int> colIndexStack = new Stack<int>();
    let mut row_index_stack: Vec<i32> = Vec::new(); // Explicitly typed for clarity
    let mut col_index_stack: Vec<i32> = Vec::new(); // Explicitly typed for clarity

    // Top element indicates candidate digits (those with False) for (row, col)
    //Stack<bool[]> usedDigitsStack = new Stack<bool[]>();
    let mut used_digits_stack: Vec<[bool; 9*9]> = Vec::new();

    // Top element is the value that was set on (row, col)
    //Stack<int> lastDigitStack = new Stack<int>();
    let mut last_digit_stack:Vec<i32> = Vec::new();

    // Indicates operation to perform next
    // - expand - finds next empty cell and puts new state on stacks
    // - move - finds next candidate number at current pos and applies it to current state
    // - collapse - pops current state from stack as it did not yield a solution
    let command : &str = "expand";
    println!("state_stack.len() ={}", state_stack.len());
    while state_stack.len() <= 9 * 9
    {
        println!("before if");
        if command == "expand"
        {
            println!("in if");
            let mut current_state : [i32; 81] = [0;81];
            if state_stack.len() > 0
            {
                // source array is state_stack.last(), destination is current_state, length is 81
                //Array.Copy(state_stack.last(), current_state, current_state.len());
                let opt_owned = state_stack.last().cloned();
                current_state = opt_owned.unwrap();
            }

            let mut best_row: i32 = -1;
            let mut best_col: i32  = -1;
            let mut best_used_digits : Vec<bool> = [false; 9*9].to_vec();
            let mut best_candidates_count : i32 = -1;
            let mut best_random_value : i32 = -1;
            let mut contains_unsolvable_cells: bool = false;
            for index in 0..81
            {
                if current_state[index] == 0
                {
                    let row: i32 = (index / 9) as i32;
                    let col: i32 = (index % 9) as i32;
                    let block_row: i32 = row / 3;
                    let block_col: i32 = col / 3;

                    let mut is_digit_used: [bool; 9] = [false; 9];
                    is_digit_used[5] = true;
                    for i in 0..9
                    {
                        let row_digit = current_state[9 * i + col as usize];
                        if row_digit > 0
                        {
                            is_digit_used[row_digit as usize - 1] = true;
                        }

                        let col_digit = current_state[9 * row as usize + i];
                        if col_digit > 0
                        {
                            is_digit_used[col_digit as usize - 1] = true;
                        }

                        let block_digit = current_state[(block_row as usize * 3 + i / 3) * 9 + (block_col as usize * 3 + i % 3)];
                        if block_digit > 0
                        {
                            is_digit_used[block_digit as usize - 1] = true;
                        }
                        let candidates_count : i32 = is_digit_used.iter() // Get an iterator over the elements
                            .filter(|&&value| !value) // Filter for false values
                            .count() as i32; // Count the remaining elements
                        //let candidates_count = is_digit_used.Where(used => !used).Count();

                        if (candidates_count == 0)
                        {
                            contains_unsolvable_cells = true;
                            break;
                        }

                        let random_value = rng.random::<i32>();

                        if best_candidates_count < 0 ||
                            candidates_count < best_candidates_count ||
                            (candidates_count == best_candidates_count && random_value < best_random_value)
                        {
                            best_row = row;
                            best_col = col;
                            best_used_digits = is_digit_used;
                            best_candidates_count = candidates_count;
                            best_random_value = random_value;
                        }
                    } // for (i = 0..8)
                }

                if (!contains_unsolvable_cells)
                {
                    state_stack.push(current_state);
                    row_index_stack.push(best_row);
                    col_index_stack.push(best_col);
                    used_digits_stack.push(best_used_digits);
                    last_digit_stack.push(0); // No digit was tried at this position
                }

                // Always try to move after expand
                command = "move";
            }

        }
    }
}