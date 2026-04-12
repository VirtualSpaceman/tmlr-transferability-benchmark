// Quick and dirty script to count the number of tuples used as input for each prediction task
// Not part of the official workflow
import groovy.json.JsonSlurper

params.input_file = "${moduleDir}/../inputs/transf_scores.csv"
params.python_path = "${moduleDir}/../"
params.conda_env = "stan"
params.results_dir = "${launchDir}/predictions"
params.tail_prob = 0.05
params.launch = false

params.compiled_csv = "${launchDir}/count_tuples.csv"

// The treatments will be combined with the scorers such that the intersection between the combinations of scorers and treatments is not empty
// The reference implementation is represented by 2, the accretions by 3 and the ablations by 4
params.bayesian_treatments = [
    // Reference implementation
    [combination: [1,2],   treatment: 'bayesian hierarchical 3-tiers',             prefix: 'bh3',     control: '',  directive: '', train_dbs: 'ALL'],
    // Accretions
    [combination: [3],     treatment: 'bayesian hierarchical 3-tiers (oracle)',    prefix: 'bh3o',    control: '',  directive: '', train_dbs: 'ORACLE'],
    // Single-dataset ablations
    [combination: [6],     treatment: 'bayesian hierarchical 3-tiers caltech101',  prefix: 'bh3DBc',  control: '',  directive: '', train_dbs: 'caltech101'],
    [combination: [6],     treatment: 'bayesian hierarchical 3-tiers flowers102',  prefix: 'bh3DBf',  control: '',  directive: '', train_dbs: 'flowers102'],
    [combination: [6],     treatment: 'bayesian hierarchical 3-tiers skin_splits', prefix: 'bh3DBs',  control: '',  directive: '', train_dbs: 'skin_splits'],
]

// The single scorers have combination: [0]
// The single scorers+imagenet have combination: [1]
params.combined_scorers = [
    // Reference implementation
    [combination: [2,3,4,5,6], scorers:'etran_energy_score ncti_score pactran_score', label: 'BEST' ],
    [combination: [2,3,4,5,6], scorers:'imagenet etran_energy_score ncti_score pactran_score', label: 'I+BEST' ],
    // Ablations
    [combination: [2,5], scorers:'leep_score nleep_score parc_score sfda_score logme_score gbc_score', label: 'MID' ],
    [combination: [2,5], scorers:'tmi_score hscore_score reg_hscore_score', label: 'WORST' ],
    [combination: [2,5], scorers:'imagenet tmi_score hscore_score reg_hscore_score', label: 'I+WORST' ],
]

// The cpus directives are a rough measure of computation exertion, not an actual measure of threads/processes

input_file_date = file(params.input_file).lastModified()
def output_ready(path, check_date) {
    f = file(path)
    if (!f.exists()) {
        return false
    }
    if (f.isEmpty()) {
        return false
    }
    if (f.lastModified() <= input_file_date) {
        return false
    }
    return true
}


process count_input_tuples {
    errorStrategy { params.launch ? 'ignore' : 'terminate' }
    input:
        tuple val(scorers),
              val(scorers_label),
              val(prefix),
              val(control),
              val(directive),
              val(train_dbs),
              val(test_db)
    output:
        val(treatment)
    exec:
        out_file   = "${prefix}_${scorers_label}_${test_db}.json"
        pred_file  = "${prefix}_${scorers_label}_${test_db}.csv"
        log_file   = "${prefix}_${scorers_label}_${test_db}.log"
        pred_label = "${prefix}_${scorers_label}"
        extra_args = train_dbs == test_db ? "--force" : ""
        json_data = new JsonSlurper().parse(file("${params.results_dir}/${out_file}").newReader())
        input_count = json_data.input.N
        treatment = [prefix: prefix, scorers_label: scorers_label, test_db: test_db, input_count: input_count]
}


process writeCsv {
    errorStrategy { params.launch ? 'retry' : 'terminate' }
    maxRetries 3
    input:
        val all_treatments
    exec:
        file(params.compiled_csv).withWriter { writer ->
            writer.writeLine("prefix,scorers_label,input_count")
            all_treatments.each { t ->
                writer.writeLine("${t.prefix},${t.scorers_label},${t.input_count}")
            }
        }
}


workflow {

    // Prepare channels with the factors

    Channel.fromList(params.bayesian_treatments)
        .set { all_treatments }

    Channel.fromPath(params.input_file)
        .splitCsv(header:true)
        .map { row -> row.dataset }
        .unique()
        .take(2)
        .map { test_db -> [test_db: test_db] }
        .set { all_test_dbs }

    Channel.fromList(params.combined_scorers)
        .set { selected_combinations }

    Channel.fromPath(params.input_file)
        .splitCsv(header:true)
        .map { row -> row.transf_metric }
        .unique()
        .take(3)
        .map { scorer -> [combination: [0], scorers: scorer, label: scorer] }
        .set { all_single_scorers }

    all_single_scorers
        .filter { it.scorers != 'imagenet' }
        .map { scorer -> [combination: [1], scorers: "imagenet ${scorer.scorers}", label: "I+${scorer.label}"]  }
        .set { all_imagenet_plus_single_scorers }

    all_imagenet_plus_single_scorers.mix(selected_combinations)
        .set { all_combined_scorers }

    all_single_scorers.mix(all_combined_scorers)
        .set { all_scorers }

    // Combine channels into configurations

    // ... For single scorers, run all treatments
    all_treatments
        | combine(all_scorers)
        | map { treatment, scorer -> [ treatment,
                                       scorer,
                                       [combination: treatment.combination.findAll {it in scorer.combination}] ] } // Adds map with the intersection of the combinations
        | filter { treatment, scorer, combined -> !combined.combination.isEmpty() } // Filters out incompatible combinations
        | combine(all_test_dbs)
        | map { it -> it.inject([:]) { acc, v -> acc + v } } // Accumulate all maps into a single one, inject is Groovy's reduce
        | set { all_configurations }

    // Run all tasks

    all_configurations
        | map { config -> tuple(config.scorers, config.label, config.prefix, config.control, config.directive, config.train_dbs, config.test_db) }
        // | randomSample(5) // Comment this like to run the whole workflow
        // | view()
        | count_input_tuples

    count_input_tuples.out
        .collect()
        | writeCsv
}
