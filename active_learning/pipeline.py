from active_learning.entropy_sampling import entropy_filter

from active_learning.embedding_diversity import diversity_sampling


def active_learning_pipeline(

    predictions,

    entropy_threshold=0.5,

    n_clusters=10
):

    print("\nSTEP 1: Entropy Sampling")

    uncertain = entropy_filter(

        predictions,

        threshold=entropy_threshold
    )

    print(f"Uncertain Samples: {len(uncertain)}")


    print("\nSTEP 2: Semantic Diversity Sampling")

    diverse = diversity_sampling(

        uncertain,

        n_clusters=n_clusters
    )

    print(f"Final Human Review Samples: {len(diverse)}")


    return diverse