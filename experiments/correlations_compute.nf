// Compute the benchmark outcomes for all the transfer scorers (including Back to Bayes)

params.input_file = "${moduleDir}/../inputs/transf_scores.csv"
params.results_dir = "${launchDir}/results_sota"

// params.input_file = "${moduleDir}/../inputs/transf_scores_frozen.csv"
// params.results_dir = "${launchDir}/frozen_results_sota"

// params.results_dir = "${launchDir}/frozen_results_linear_ablations_mid"
// params.input_file = "${moduleDir}/../inputs/frozen_linear_ablations_mid.csv"

// params.input_file = "${moduleDir}/../inputs/linear_ablations_mid.csv"
// params.results_dir = "${launchDir}/results_linear_ablations_mid"

params.python_path = "${moduleDir}/../"
params.conda_env = "stan"

params.tail_prob = 0.05
params.launch = false


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

process compute {
    publishDir "${params.results_dir}"
    cpus 2
    errorStrategy { params.launch ? 'retry' : 'terminate' }
    maxRetries 3
    input:
        val(scorer)
    output:
        path("${out_file}"), emit: out_json
        path("${log_file}"), emit: log
    when:
        !file("${params.results_dir}/correlations_${scorer}.json").exists()
    shell:
        out_file = "correlations_${scorer}.json"
        log_file = "correlations_${scorer}.log"
        '''
        eval "$(conda shell.bash hook)"
        conda activate !{params.conda_env}
        export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}!{params.python_path}"

        python -m analysis.correlations_compute \
            --significance_level !{params.tail_prob} \
            --output_file '!{out_file}' \
            --input_file '!{params.input_file}' \
            --scorer '!{scorer}' 2>&1 | tee '!{log_file}'
        '''
}

workflow {

    // Prepare channels with the factors

    Channel.fromPath(params.input_file)
        .splitCsv(header:true)
        .map { row -> row.transf_metric }
        .unique()
        // .take(10) // Comment this like to run the whole workflow
        .set { all_scorers }

    // Perform tasks

    // ... for each scorer, run the classical and rank regressions
    all_scorers
        // .view()
        | compute

}
