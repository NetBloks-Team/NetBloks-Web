<script context="module" lang="ts">
    // Placeholder for module script
    let count = 0;
</script>

<p>You have clicked {count} times</p>
<button
    on:click={() => {
        count++;
        console.log(count);
    }}>Click me!</button
>
