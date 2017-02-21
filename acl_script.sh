#! /bin/bash

main_directory="aclImdb"

testing_set="training_movie_data.csv"

output="new_reviews.csv"

total_reviews=0
found_reviews=0
new_reviews=0

echo "review,sentiment" > $output

for directory in $main_directory/*; do
  if [[ -d $directory ]]; then
    for sub_directory in $directory/*; do
      if [[ -d $sub_directory ]]; then
        base_name="$(basename $sub_directory)"

        #Determine if these are positive or negative reviews
        if [[ "$base_name" = "neg" ]]; then
          sentiment=0
        elif [[ "$base_name" = "pos" ]]; then
          sentiment=1
        else
          sentiment=-1
        fi

        if ((sentiment >= 0)); then
          for review in $sub_directory/*; do
            ((total_reviews+=1))
            if grep -q -s -f $review $testing_set; then
              ((found_reviews+=1))
            else
              ((new_reviews+=1))
              printf "\"%s\",%d\n" "$(cat $review)" $sentiment >> $output
            fi
            printf "\rTotal reviews found: %5d\tReviews in test set: %5d\tNew reviews: %5d" $total_reviews $found_reviews $new_reviews
          done
        fi

      fi
    done
  fi
done
